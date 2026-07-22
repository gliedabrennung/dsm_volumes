from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as ndi_shift

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsm_volumes import coregistration, multitemporal
from dsm_volumes import datum as datum_mod
from dsm_volumes import volumes as volumes_mod
from dsm_volumes.config import PipelineConfig
from dsm_volumes.pipeline import VolumeCalculationPipeline
from dsm_volumes.reporting import RasterTransform
from dsm_volumes.segmentation import ObjectCandidate
from examples.synthetic_scene import generate_extreme_terrain_patch, generate_synthetic_scene

OUT_DIR = Path(__file__).parent / "example_outputs"
OUT_DIR.mkdir(exist_ok=True)


def hillshade(dsm: np.ndarray, pixel_size_m: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    gy, gx = np.gradient(dsm, pixel_size_m)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(shaded, 0, 1)


def section(title: str) -> None:
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def make_candidate(mask: np.ndarray, obj_id: int, kind: str) -> ObjectCandidate:
    ys, xs = np.nonzero(mask)
    slice_ = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
    return ObjectCandidate(id=obj_id, kind=kind, mask=mask, slice_=slice_, area_px=int(mask.sum()))


def find_match(vol_results, obj, pixel_size_m: float):
    row_true, col_true = obj.cy_m / pixel_size_m, obj.cx_m / pixel_size_m
    best, best_d = None, float("inf")
    for v in vol_results:
        d = (v.centroid_row - row_true) ** 2 + (v.centroid_col - col_true) ** 2
        if d < best_d:
            best, best_d = v, d
    return best


def main() -> None:
    section("1. Синтетическая сцена (для численной валидации точности алгоритмов)")
    scene = generate_synthetic_scene()
    print(f"Растр: {scene.nx}x{scene.ny} px, пиксель {scene.pixel_size_m} м -> "
          f"{scene.extent_m[0]:.0f}x{scene.extent_m[1]:.0f} м")
    print("Рельеф основания: уклон 3%/2% + плавное синусоидальное всхолмление (A=0.12 м) + шум (A~0.08 м) "
          "— намеренно НЕ плоский (амплитуда откалибрована так, чтобы порог детектирования "
          "можно было снизить без ложных срабатываний на фоне, см. раздел 3).")
    for name, obj in scene.objects_t1.items():
        print(f"  {name}: R={obj.radius_m:.1f} м, H={obj.height_m:+.2f} м -> "
              f"истинный объём = {obj.true_volume_m3():.1f} м³")
    print(f"  vehicle_D (имитация техники): {scene.vehicle_box['w_m']}x{scene.vehicle_box['l_m']} м, "
          f"h={scene.vehicle_box['height_m']} м -> объём = "
          f"{scene.vehicle_box['w_m']*scene.vehicle_box['l_m']*scene.vehicle_box['height_m']:.1f} м³ "
          "(должен быть ОТСЕЯН, не должен попасть в ведомость)")

    plt.imsave(OUT_DIR / "input_ortho_t1.png", scene.ortho_t1)
    plt.imsave(OUT_DIR / "input_hillshade_t1.png", hillshade(scene.dsm_t1, scene.pixel_size_m), cmap="gray")
    print("\nСохранены входные растры для наглядности: input_ortho_t1.png, input_hillshade_t1.png")

    section("2. Точность восстановления датума на сложном рельефе (pile_A)")
    config = PipelineConfig(pixel_size_m=scene.pixel_size_m)
    X, Y = np.meshgrid(np.arange(scene.nx) * scene.pixel_size_m, np.arange(scene.ny) * scene.pixel_size_m)
    pile_A, pile_B, pit_C = scene.objects_t1["pile_A"], scene.objects_t1["pile_B"], scene.objects_t1["pit_C"]

    def true_disk_mask(obj):
        return ((X - obj.cx_m) ** 2 + (Y - obj.cy_m) ** 2) <= obj.radius_m**2

    true_mask_A, true_mask_B, true_mask_C = true_disk_mask(pile_A), true_disk_mask(pile_B), true_disk_mask(pit_C)
    all_true_mask = true_mask_A | true_mask_B | true_mask_C
    cand_true_A = make_candidate(true_mask_A, 901, "fill")

    print("(используется ИСТИННЫЙ контур объекта — изолируем точность датума от точности сегментации)\n")
    print(f"{'метод датума':<14}{'расч. объём, м³':>18}{'истинный объём, м³':>21}{'ошибка':>10}")
    for method in ("constant", "plane", "tin"):
        datum_local, window = datum_mod.compute_datum_for_object(scene.dsm_t1, all_true_mask, cand_true_A, config, method=method)
        dsm_local = scene.dsm_t1[window]
        mask_local = cand_true_A.mask[window]
        vr = volumes_mod.compute_object_volume(dsm_local, datum_local, mask_local, scene.pixel_size_m**2, 901, "fill")
        err_pct = (vr.fill_volume_m3 - pile_A.true_volume_m3()) / pile_A.true_volume_m3() * 100
        print(f"{method:<14}{vr.fill_volume_m3:>18.1f}{pile_A.true_volume_m3():>21.1f}{err_pct:>+9.1f}%")
    print("\n'constant' и 'plane' — типичные упрощённые подходы (одна отметка/плоскость по обмеру периметра);")
    print("'tin' — предлагаемый метод (интерполяция по кольцу точек вокруг объекта).")

    section("2b. Стресс-тест датума: рельеф, искривлённый уже В ПРЕДЕЛАХ объекта")
    terrain_ex, X_ex, Y_ex = generate_extreme_terrain_patch(pixel_size_m=scene.pixel_size_m)
    pile_ex_R, pile_ex_H = 7.0, 3.5
    cx_ex, cy_ex = terrain_ex.shape[1] * scene.pixel_size_m / 2, terrain_ex.shape[0] * scene.pixel_size_m / 2
    r2_ex = (X_ex - cx_ex) ** 2 + (Y_ex - cy_ex) ** 2
    profile_ex = np.where(r2_ex <= pile_ex_R**2, pile_ex_H * (1 - r2_ex / pile_ex_R**2), 0.0)
    dsm_ex = terrain_ex + profile_ex
    true_v_ex = 0.5 * np.pi * pile_ex_H * pile_ex_R**2
    mask_ex = r2_ex <= pile_ex_R**2
    cand_ex = make_candidate(mask_ex, 902, "fill")

    print(f"Уклон 9%/6% + всхолмление с длиной волны (18x22 м), сопоставимой с диаметром объекта "
          f"(2R={2*pile_ex_R:.0f} м) — плоскость принципиально не может описать такой рельеф.\n")
    print(f"{'метод датума':<14}{'расч. объём, м³':>18}{'истинный объём, м³':>21}{'ошибка':>10}")
    for method in ("constant", "plane", "tin"):
        datum_local, window = datum_mod.compute_datum_for_object(dsm_ex, mask_ex, cand_ex, config, method=method)
        dsm_local = dsm_ex[window]
        mask_local = cand_ex.mask[window]
        vr = volumes_mod.compute_object_volume(dsm_local, datum_local, mask_local, scene.pixel_size_m**2, 902, "fill")
        err_pct = (vr.fill_volume_m3 - true_v_ex) / true_v_ex * 100
        print(f"{method:<14}{vr.fill_volume_m3:>18.1f}{true_v_ex:>21.1f}{err_pct:>+9.1f}%")
    print("\nНа рельефе, искривлённом в масштабе самого объекта, 'plane' тоже начинает заметно ошибаться —")
    print("только локальная TIN-интерполяция по кольцу остаётся точной вне зависимости от формы основания.")

    # ------------------------------------------------------------------
    section("3. Диагностика систематической ошибки и её устранение")
    print("Первая версия конвейера показывала завышенную ошибку (-6...-17% на разных объектах),")
    print("сильно превышающую то, что объясняется одним лишь усечением контура по порогу высоты.")
    print("Диагностика (см. историю разработки): восстановленный TIN-датум оказался смещён на")
    print("~+0.13 м относительно истинного рельефа — потому что кольцо точек 'грунта' бралось сразу")
    print("от границы маски, а сама граница (проведённая по порогу высоты) всё ещё немного заходит")
    print("на 'юбку' объекта — эти точки просачивались в кольцо и смещали датум.")
    print("Исправление: добавлен защитный отступ datum_safety_margin_m — граница исключаемой зоны")
    print("расширяется на этот отступ ПЕРЕД тем, как строится кольцо сбора точек грунта.\n")

    truth_by_name = {"pile_A": pile_A.true_volume_m3(), "pile_B": pile_B.true_volume_m3(), "pit_C": pit_C.true_volume_m3()}
    print(f"{'защитный отступ, м':<20}{'pile_A':>10}{'pile_B':>10}{'pit_C':>10}")
    for margin in (0.0, 0.3, 0.5, 0.7):
        cfg_m = PipelineConfig(pixel_size_m=scene.pixel_size_m, fill_threshold_m=config.fill_threshold_m,
                                cut_threshold_m=config.cut_threshold_m, datum_safety_margin_m=margin)
        res_m = VolumeCalculationPipeline(cfg_m).run_single_epoch(scene.dsm_t1, scene.ortho_t1)
        errs = {}
        for name, obj in scene.objects_t1.items():
            m = find_match(res_m.volumes, obj, scene.pixel_size_m)
            calc = m.fill_volume_m3 if obj.height_m > 0 else m.cut_volume_m3
            errs[name] = (calc - truth_by_name[name]) / truth_by_name[name] * 100
        marker = "  <- было (без отступа)" if margin == 0.0 else ("  <- рабочее значение" if margin == config.datum_safety_margin_m else "")
        print(f"{margin:<20.2f}{errs['pile_A']:>+9.1f}%{errs['pile_B']:>+9.1f}%{errs['pit_C']:>+9.1f}%{marker}")

    print(f"\nОтдельно: порог детектирования (fill/cut_threshold_m = {config.fill_threshold_m} м) — тоже")
    print("калибруемый параметр с trade-off (выше -> меньше ложных срабатываний, но больше отсечённой")
    print("'юбки'; ниже -> наоборот), но после исправления датума его влияние на итоговую точность")
    print("гораздо меньше, чем было смещение датума:")
    print(f"{'порог, м':<10}{'найдено объектов':>18}{'pile_A':>10}{'pile_B':>10}{'pit_C':>10}")
    for thr in (0.30, 0.24, 0.18, 0.12):
        cfg_thr = PipelineConfig(pixel_size_m=scene.pixel_size_m, fill_threshold_m=thr, cut_threshold_m=thr)
        res_thr = VolumeCalculationPipeline(cfg_thr).run_single_epoch(scene.dsm_t1, scene.ortho_t1)
        errs = {}
        for name, obj in scene.objects_t1.items():
            m = find_match(res_thr.volumes, obj, scene.pixel_size_m)
            calc = m.fill_volume_m3 if obj.height_m > 0 else m.cut_volume_m3
            errs[name] = (calc - truth_by_name[name]) / truth_by_name[name] * 100
        flag = "" if len(res_thr.volumes) == 3 else "  <- ложные срабатывания"
        print(f"{thr:<10.2f}{len(res_thr.volumes):>18}{errs['pile_A']:>+9.1f}%{errs['pile_B']:>+9.1f}%{errs['pit_C']:>+9.1f}%{flag}")
    print(f"\nРабочие значения конвейера по умолчанию: fill/cut_threshold_m = {config.fill_threshold_m} м, "
          f"datum_safety_margin_m = {config.datum_safety_margin_m} м (для реальной площадки калибруются "
          "по остатку dsm-trend и по типичной высоте объектов этой площадки).")

    section("4. Полный конвейер, эпоха t1: детектирование + фильтрация + объём")
    pipeline = VolumeCalculationPipeline(config)
    transform = RasterTransform(origin_x=500_000.0, origin_y=6_000_000.0, pixel_size_m=scene.pixel_size_m)
    result_t1 = pipeline.run_single_epoch(
        scene.dsm_t1, scene.ortho_t1, out_dir=OUT_DIR, transform=transform, report_prefix="epoch_t1"
    )

    n_rejected = sum(1 for c in result_t1.rejected if not c.is_valid_object)
    print(f"Кандидатов найдено детектором: {len(result_t1.rejected)}; "
          f"прошло фильтр шума: {len(result_t1.volumes)}; отсеяно: {n_rejected}\n")
    for c in result_t1.rejected:
        status = "включён" if c.is_valid_object else "ОТСЕЯН "
        print(f"  кандидат #{c.object_id}: площадь={c.area_m2:7.1f} м², extent={c.extent:.2f}, "
              f"цвет-техники={c.machinery_color_fraction:4.0%} -> {status} | {c.reason}")

    print(f"\n{'объект':<10}{'детект.ID':>10}{'площадь,м²':>13}{'расчёт,м³':>13}{'истина,м³':>13}{'ошибка':>10}")
    for name, obj in scene.objects_t1.items():
        match = find_match(result_t1.volumes, obj, scene.pixel_size_m)
        true_v = obj.true_volume_m3()
        calc_v = match.fill_volume_m3 if obj.height_m > 0 else match.cut_volume_m3
        err_pct = (calc_v - true_v) / true_v * 100
        print(f"{name:<10}{match.object_id:>10}{match.area_m2:>13.1f}{calc_v:>13.1f}{true_v:>13.1f}{err_pct:>+9.1f}%")

    section("5. Мультивременное сравнение (DoD): эпоха t0 -> t1")
    result_multi = pipeline.run_multitemporal(
        scene.dsm_t0, scene.dsm_t1, scene.ortho_t1, out_dir=OUT_DIR, transform=transform, report_prefix="multitemporal"
    )
    site = result_multi.output_files.get("whole_site_summary", {})
    vehicle_v = scene.vehicle_box["w_m"] * scene.vehicle_box["l_m"] * scene.vehicle_box["height_m"]

    print(f"Изменение по ВСЕЙ площадке без учёта фильтрации техники (сырой DoD):")
    print(f"  насыпь = {site.get('fill_volume_m3', 0):.1f} м³   выемка = {site.get('cut_volume_m3', 0):.1f} м³")

    fill_sum_valid = sum(v.fill_volume_m3 for v in result_multi.volumes if v.kind == "fill")
    print(f"Сумма насыпи ТОЛЬКО по объектам, прошедшим фильтр шума: {fill_sum_valid:.1f} м³")
    print(f"Разница (сырой DoD минус отфильтрованная сумма): {site.get('fill_volume_m3', 0) - fill_sum_valid:.1f} м³ "
          f"(ожидается ≈ объём техники = {vehicle_v:.1f} м³ — она видна в сыром DoD, но не в ведомости)")

    print(f"\n{'объект':<10}{'детект.ID':>10}{'ΔV расч.,м³':>15}{'ΔV истин.,м³':>15}{'ошибка':>10}")
    expected_delta = {
        "pile_A": pile_A.true_volume_m3() - scene.objects_t0["pile_A"].true_volume_m3(),
        "pile_B": pile_B.true_volume_m3(),
        "pit_C": pit_C.true_volume_m3() - scene.objects_t0["pit_C"].true_volume_m3(),
    }
    for name, obj in scene.objects_t1.items():
        match = find_match(result_multi.volumes, obj, scene.pixel_size_m)
        true_delta = expected_delta[name]
        calc_delta = match.net_volume_m3 if obj.height_m > 0 else -match.net_volume_m3
        err_pct = (calc_delta - true_delta) / true_delta * 100 if abs(true_delta) > 1e-6 else 0.0
        print(f"{name:<10}{match.object_id:>10}{calc_delta:>15.1f}{true_delta:>15.1f}{err_pct:>+9.1f}%")

    print(f"\nОбъектов в мультивременной ведомости: {len(result_multi.volumes)} "
          f"({'техника корректно исключена' if len(result_multi.volumes) == 3 else 'ТРЕБУЕТСЯ ПРОВЕРКА'})")

    # ------------------------------------------------------------------
    section("6. Co-регистрация эпох при рассогласованной геопривязке")
    print("Реалистичная проблема: между двумя съёмками georeference слегка 'поплыл'")
    print("(разные схемы GCP, дрейф RTK/PPK) — на ЦМП это выглядит как горизонтальный сдвиг")
    print("в доли/единицы пикселей. Без коррекции такой сдвиг на любой границе перепада высоты")
    print("создаёт ЛОЖНЫЙ сигнал в DoD (насыпь с одного края объекта, выемка с другого),")
    print("не имеющий отношения к реальному изменению объёма.\n")

    true_shift_px = (1.2, -0.8)
    dsm_t0_misaligned = ndi_shift(scene.dsm_t0, [-s for s in true_shift_px], order=3, mode="nearest")

    whole_baseline = multitemporal.compute_change_volume(
        multitemporal.compute_dod(scene.dsm_t0, scene.dsm_t1), scene.pixel_size_m
    )
    whole_no_fix = multitemporal.compute_change_volume(
        multitemporal.compute_dod(dsm_t0_misaligned, scene.dsm_t1), scene.pixel_size_m
    )
    coreg = coregistration.coregister(dsm_t0_misaligned, scene.dsm_t1)
    whole_fixed = multitemporal.compute_change_volume(
        multitemporal.compute_dod(dsm_t0_misaligned, coreg.dsm_aligned), scene.pixel_size_m
    )

    print(f"Внесённый (искусственный) сдвиг: {true_shift_px} px. Определено фазовой корреляцией: "
          f"({coreg.shift_row_px:+.2f}, {coreg.shift_col_px:+.2f}) px (ожидается ≈ {tuple(-s for s in true_shift_px)}).\n")
    print(f"{'вариант':<32}{'насыпь по площадке, м³':>24}{'ошибка vs эталон':>18}")
    err_no = abs(whole_no_fix.fill_volume_m3 - whole_baseline.fill_volume_m3) / whole_baseline.fill_volume_m3 * 100
    err_fixed = abs(whole_fixed.fill_volume_m3 - whole_baseline.fill_volume_m3) / whole_baseline.fill_volume_m3 * 100
    print(f"{'эталон (сдвига нет)':<32}{whole_baseline.fill_volume_m3:>24.1f}{'—':>18}")
    print(f"{'сдвиг есть, БЕЗ коррекции':<32}{whole_no_fix.fill_volume_m3:>24.1f}{err_no:>+17.1f}%")
    print(f"{'сдвиг есть, С коррекцией':<32}{whole_fixed.fill_volume_m3:>24.1f}{err_fixed:>+17.1f}%")
    print("\nВключается параметром run_multitemporal(auto_coregister=True) / CLI --auto-coregister.")
    print("Ограничение метода (честно): работает по перепадам высоты самих объектов — на участке")
    print("АБСОЛЮТНО без объектов (только гладкий фон) фазовая корреляция вырождается в нулевой сдвиг;")
    print("для таких площадок нужна регистрация по ортофото или по GCP/чек-пойнтам, а не по ЦМП.")

    section("7. Файлы отчётов")
    for label, files in (("Эпоха t1", result_t1.output_files), ("Мультивременной", result_multi.output_files)):
        print(f"[{label}]")
        for k, v in files.items():
            if isinstance(v, dict):
                continue
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
