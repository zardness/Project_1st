#!/usr/bin/env python3
"""
지역구 날씨별 이용량 EDA
------------------------------------------------------------
입력
--usage : 대여 이력 CSV (예: 2001_merge.csv ... 2512_merge.csv, 대여일시/대여자치구 컬럼 필요)
--rain  : 일일강수량_YYYY.csv  (날짜, 자치구, 일강수량(mm))
--temp  : 일일기온_YYYY.csv    (날짜, 자치구, 평균기온(℃), 최저기온(℃), 최고기온(℃))
--air   : 일별평균대기오염도_YYYY.csv (측정일시, 측정소명, 미세먼지농도, 초미세먼지농도)
출력(--out)
daily_gu_weather.csv   : 자치구×일 분석용 통합 테이블 (모델링 입력으로도 사용)
data_quality_by_gu.csv : 구별 결측 현황
regression_summary.csv : 날씨 변수의 순수 효과(요일·공휴일·구·월 통제)
gu_sensitivity.csv     : 구별 기온/강수 민감도
01~07_*.png            : 시각화

사용 예
python weather_usage_eda.py --usage "data/usage/**/*_merge.csv" \
    --rain "data/rain/일일강수량_*.csv" --temp "data/temp/일일기온_*.csv" \
    --air "data/air/일별평균대기오염도_*.csv" --out result
"""
import argparse
import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- 설정
TEMP_EDGES = [-30, -5, 0, 5, 10, 15, 20, 25, 30, 50]
TEMP_LABELS = ["<-5", "-5~0", "0~5", "5~10", "10~15", "15~20", "20~25", "25~30", "30+"]
RAIN_EDGES = [-0.01, 0.09, 5, 20, 1000]
RAIN_LABELS = ["무강수", "약(0.1~5)", "보통(5~20)", "강(20+)"]
# 환경부 일평균 기준(㎍/㎥)
PM10_EDGES, PM10_LABELS = [-1, 30, 80, 150, 1000], ["좋음", "보통", "나쁨", "매우나쁨"]
PM25_EDGES, PM25_LABELS = [-1, 15, 35, 75, 1000], ["좋음", "보통", "나쁨", "매우나쁨"]
EXCL = set()
SUSPECT_RATIO = 0.3     # 같은 유형(평일/휴일)의 중앙값 대비 이 비율 미만이면 '의심일'로 제외
MIN_N = 3               # 히트맵/민감도에서 셀당 최소 관측 수

# 'holidays' 패키지가 없을 때 쓰는 2020년 공휴일(대체·임시공휴일 포함)
FALLBACK_HOLIDAYS_2020 = [
    "2020-01-01", "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27",
    "2020-04-15", "2020-04-30", "2020-05-05", "2020-06-06", "2020-08-15",
    "2020-08-17", "2020-09-30", "2020-10-01", "2020-10-02", "2020-10-03",
    "2020-10-09", "2020-12-25",
]


def setup_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR",
                "Noto Sans CJK JP", "Noto Sans CJK SC"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"][0])


def read_csv_any(path, **kw):
    last = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError as e:
            last = e
    raise last


def load_glob(pattern, **kw):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        sys.exit(f"[오류] 파일을 찾지 못했습니다: {pattern}")
    return pd.concat([read_csv_any(f, **kw) for f in files], ignore_index=True)


# ----------------------------------------------------------------- 로딩
def load_usage_daily(pattern):
    """대여 이력(수십만 행/월)을 파일 단위로 읽어 '일×대여자치구' 건수로 즉시 집계."""
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        sys.exit(f"[오류] 이용 파일을 찾지 못했습니다: {pattern}")
    parts = []
    for f in files:
        d = read_csv_any(f, usecols=["대여일시", "대여자치구"])
        d["date"] = pd.to_datetime(d["대여일시"], errors="coerce").dt.normalize()
        bad = d["date"].isna().sum()
        if bad:
            print(f"  [경고] {os.path.basename(f)}: 날짜 파싱 실패 {bad}행 제외")
        parts.append(d.dropna(subset=["date"]).groupby(["date", "대여자치구"]).size()
                    .rename("rentals").reset_index())
        print(f"  읽음: {os.path.basename(f)} ({len(d):,}행)")
    u = pd.concat(parts).groupby(["date", "대여자치구"], as_index=False)["rentals"].sum()
    return u.rename(columns={"대여자치구": "gu"})


def get_holidays(years, extra_file=None):
    """공휴일(대체·임시공휴일 포함). `pip install -U holidays` 필요. 여러 해를 분석할 때 패키지가 없으면 중단한다."""
    try:
        import holidays
        hol = {pd.Timestamp(d) for d in holidays.KR(years=years).keys()}
    except ImportError:
        if set(years) == {2020}:
            print("  [안내] 'holidays' 패키지가 없어 내장 2020년 공휴일을 사용합니다.")
            hol = set(pd.to_datetime(FALLBACK_HOLIDAYS_2020))
        else:
            sys.exit("[오류] 'holidays' 패키지가 필요합니다 (분석 기간: "
                    f"{min(years)}~{max(years)}).\n  → pip install -U holidays  후 다시 실행하세요.\n"
                    "  (설치가 어려우면 공휴일 날짜 목록 CSV(컬럼명 date)를 --holiday-file 로 지정)")
    if extra_file:  # 패키지에 빠진 임시공휴일 등을 직접 추가
        extra = read_csv_any(extra_file)
        hol |= set(pd.to_datetime(extra["date"]))
        print(f"  공휴일 보정 파일에서 {len(extra)}일 추가")
    return hol


def build_table(args):
    print("[1] 데이터 로딩")
    usage = load_usage_daily(args.usage)
    rain = load_glob(args.rain).rename(columns={"날짜": "date", "자치구": "gu", "일강수량(mm)": "rain"})
    temp = load_glob(args.temp).rename(columns={"날짜": "date", "자치구": "gu", "평균기온(℃)": "tavg",
                                                "최저기온(℃)": "tmin", "최고기온(℃)": "tmax"})
    air = load_glob(args.air).rename(columns={"측정일시": "date", "측정소명": "gu",
                                            "미세먼지농도(㎍/㎥)": "pm10", "초미세먼지농도(㎍/㎥)": "pm25"})
    air["date"] = pd.to_datetime(air["date"].astype(str), format="%Y%m%d")
    for d in (rain, temp):
        d["date"] = pd.to_datetime(d["date"])
    rain, temp = rain[["date", "gu", "rain"]], temp[["date", "gu", "tavg", "tmin", "tmax"]]
    air = air[["date", "gu", "pm10", "pm25"]]

    # 이용 데이터에 있는 기간 × 25개 구 전체 격자(이용 0건인 구-일도 보존)
    grid = pd.MultiIndex.from_product(
        [pd.date_range(usage["date"].min(), usage["date"].max()), sorted(usage["gu"].unique())],
        names=["date", "gu"]).to_frame(index=False)
    df = grid.merge(usage, on=["date", "gu"], how="left")
    n_zero = df["rentals"].isna().sum()
    df["rentals"] = df["rentals"].fillna(0)
    if n_zero:
        print(f"  이용 기록이 없는 구-일 {n_zero}건 → 0건으로 처리 (결측 vs 실제 0 여부 확인 필요)")
    for w in (rain, temp, air):
        df = df.merge(w, on=["date", "gu"], how="left")
    return df


# ----------------------------------------------------------------- 파생변수
def add_features(df, keep_suspect=False, holiday_file=None, out=None):
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    df["ym"] = df["date"].dt.strftime("%Y-%m")
    df["dow"] = df["date"].dt.dayofweek
    hol = get_holidays(sorted(df["year"].unique()), holiday_file)
    hol_used = pd.DataFrame({"date": sorted(h for h in hol if h.year in set(df["year"]))})
    hol_used["요일"] = hol_used["date"].dt.day_name()
    print("  적용된 평일 공휴일 수(연도별):", hol_used[hol_used["date"].dt.dayofweek < 5]
          .groupby(hol_used["date"].dt.year).size().to_dict(), "← 연도별 개수가 비정상적이면 result/holidays_used.csv 확인")
    if out:
        hol_used.to_csv(os.path.join(out, "holidays_used.csv"), index=False, encoding="utf-8-sig")
    df["is_holiday"] = df["date"].isin(hol).astype(int)
    df["dayoff"] = ((df["dow"] >= 5) | (df["is_holiday"] == 1)).astype(int)

    df["rain_flag"] = (df["rain"] >= 0.1).astype(float).where(df["rain"].notna())
    df["rain_bin"] = pd.cut(df["rain"], RAIN_EDGES, labels=RAIN_LABELS)
    df["temp_bin"] = pd.cut(df["tavg"], TEMP_EDGES, labels=TEMP_LABELS)
    df["pm10_grade"] = pd.cut(df["pm10"], PM10_EDGES, labels=PM10_LABELS)
    df["pm25_grade"] = pd.cut(df["pm25"], PM25_EDGES, labels=PM25_LABELS)

    # 이용 데이터 이상일(시스템 장애/누락 의심) 탐지: 전체 합계가 같은 유형 중앙값의 30% 미만
    city = df.groupby(["date", "dayoff"])["rentals"].sum().reset_index()
    med = city.groupby("dayoff")["rentals"].transform("median")
    city["suspect"] = city["rentals"] < SUSPECT_RATIO * med
    df = df.merge(city[["date", "suspect"]], on="date", how="left")
    # 'suspect'는 탐지 결과, 'excluded'는 실제 분석 제외 여부.
    # 극단적 날씨(폭우·폭설)로 이용이 급감했을 수도 있으므로 --keep-suspect 로 포함해 비교할 수 있다.
    df["excluded"] = df["suspect"] & (not keep_suspect)

    # 구 규모·요일 유형 효과를 제거한 상대 이용량 (1.0 = 해당 구·연도·평일/휴일 평균)
    ok = ~df["excluded"]
    base = df[ok].groupby(["gu", "year", "dayoff"])["rentals"].mean().rename("base").reset_index()
    df = df.merge(base, on=["gu", "year", "dayoff"], how="left")
    df["usage_ratio"] = df["rentals"] / df["base"]
    return df.drop(columns="base")


# ----------------------------------------------------------------- 통계
def ols_cluster(y, X, cluster):
    """OLS + 날짜 단위 군집 강건 표준오차 (같은 날 구들의 오차는 상관되므로)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    e = y - X @ beta
    n, k = X.shape
    codes = pd.factorize(cluster)[0]
    G = codes.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, codes, X * e[:, None])
    meat = S.T @ S * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(bread @ meat @ bread))
    return beta, se


def regression(df):
    d = df[~df["excluded"] & (df["rentals"] > 0)].dropna(subset=["tavg", "rain_flag", "pm10"]).copy()
    d["pm10_10"] = d["pm10"] / 10
    fe = ["gu"] + (["ym"] if d["ym"].nunique() > 1 else [])
    X = pd.concat([d[["tavg", "rain_flag", "pm10_10", "dayoff"]],
                pd.get_dummies(d[fe], drop_first=True, dtype=float)], axis=1)
    X.insert(0, "const", 1.0)
    beta, se = ols_cluster(np.log(d["rentals"]), X.values, d["date"].values)
    names = {"tavg": "기온 +1℃", "rain_flag": "강수일(0.1mm 이상)", "pm10_10": "미세먼지 +10㎍/㎥",
            "dayoff": "주말·공휴일"}
    rows = []
    for c, lab in names.items():
        i = list(X.columns).index(c)
        rows.append({"변수": lab, "이용량 변화(%)": (np.exp(beta[i]) - 1) * 100,
                    "계수": beta[i], "표준오차": se[i], "t값": beta[i] / se[i]})
    out = pd.DataFrame(rows).round(3)
    out.attrs["n"] = len(d)
    return out


def gu_sensitivity(df):
    d = df[~df["excluded"] & (df["usage_ratio"] > 0)].dropna(subset=["tavg", "rain"])
    rows = []
    for gu, g in d.groupby("gu"):
        slope = np.polyfit(g["tavg"], np.log(g["usage_ratio"]), 1)[0] if g["tavg"].nunique() > 2 else np.nan
        wet, dry = g[g["rain"] >= 0.1], g[g["rain"] < 0.1]
        rain_eff = (wet["usage_ratio"].mean() / dry["usage_ratio"].mean() - 1) * 100 if len(wet) >= MIN_N and len(dry) >= MIN_N else np.nan
        rows.append({"자치구": gu, "기온 1℃당 변화(%)": (np.exp(slope) - 1) * 100,
                    "강수일 이용량 변화(%)": rain_eff, "강수일수": len(wet), "관측일수": len(g)})
    return pd.DataFrame(rows).round(2)


# ----------------------------------------------------------------- 시각화
def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)


def annotate_n(ax, order, counts):
    for i, k in enumerate(order):
        ax.text(i, ax.get_ylim()[0], f"n={counts.get(k, 0)}", ha="center", va="bottom", fontsize=8, color="gray")


def plot_all(df, sens, out):
    global EXCL
    EXCL = set(df.loc[df["excluded"], "date"])
    d = df[~df["excluded"]].copy()

    # 01 시계열: 전체 이용량 + 기온 + 강수
    city = df.groupby("date").agg(rentals=("rentals", "sum"), tavg=("tavg", "mean"),
                                rain=("rain", "mean"), dayoff=("dayoff", "max"),
                                suspect=("suspect", "max")).reset_index()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(city["date"], city["rentals"], color="#2a6fb0", lw=1.6)
    for _, r in city[city["dayoff"] == 1].iterrows():
        a1.axvspan(r["date"] - pd.Timedelta(hours=12), r["date"] + pd.Timedelta(hours=12), color="#f2c14e", alpha=.25, lw=0)
    for _, r in city[city["suspect"]].iterrows():
        a1.scatter(r["date"], r["rentals"], color="red", zorder=5)
        a1.annotate("이용량 급감일" + ("(제외)" if r["date"] in EXCL else "(포함)"), (r["date"], r["rentals"]), xytext=(5, 12), textcoords="offset points", color="red", fontsize=8)
    a1.set_ylabel("일 대여 건수 (서울 전체)")
    a1.set_title("일별 이용량과 날씨  (노란 음영 = 주말·공휴일)")
    a2.plot(city["date"], city["tavg"], color="#d1495b", lw=1.6)
    a2.set_ylabel("평균기온(℃, 25개 구 평균)", color="#d1495b")
    b = a2.twinx()
    b.bar(city["date"], city["rain"], color="#4a90c2", alpha=.6, width=0.8)
    b.set_ylabel("강수량(mm, 25개 구 평균)", color="#4a90c2")
    b.grid(False)
    save(fig, out, "01_timeline.png")

    # 02 기온 구간별
    t = d.dropna(subset=["tavg"]).groupby("temp_bin", observed=True)["usage_ratio"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(t.index.astype(str), t["mean"], color="#d1495b", alpha=.85)
    ax.axhline(1, color="k", lw=.8, ls="--")
    for i, (m, n) in enumerate(zip(t["mean"], t["count"])):
        ax.text(i, m, f"{m:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, t["mean"].max() * 1.2)
    ax.set_xlabel("평균기온 구간(℃)")
    ax.set_ylabel("상대 이용량 (1.0 = 구·평일/휴일 평균)")
    ax.set_title("기온 구간별 상대 이용량")
    save(fig, out, "02_temp_bin.png")

    # 03 강수 구간별
    r = d.dropna(subset=["rain"])
    order = [k for k in RAIN_LABELS if k in set(r["rain_bin"].astype(str))]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=r, x="rain_bin", y="usage_ratio", order=order, color="#4a90c2", ax=ax, fliersize=2)
    cnt, med = r["rain_bin"].astype(str).value_counts(), r.groupby(r["rain_bin"].astype(str))["usage_ratio"].median()
    for i, k in enumerate(order):
        txt = f"n={cnt.get(k, 0)}" + ("" if k == order[0] else f"\n{(med[k] / med[order[0]] - 1) * 100:+.0f}%")
        ax.text(i, ax.get_ylim()[1] * 0.97, txt, ha="center", va="top", fontsize=9)
    ax.set_xlabel("일강수량 구간(mm)")
    ax.set_ylabel("상대 이용량")
    ax.set_title("강수량 구간별 상대 이용량  (% = 무강수 대비 중앙값 변화)")
    save(fig, out, "03_rain_bin.png")

    # 04 대기오염 등급별
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, col, lab in [(axes[0], "pm10_grade", "미세먼지(PM10) 등급"), (axes[1], "pm25_grade", "초미세먼지(PM2.5) 등급")]:
        s = d.dropna(subset=[col])
        order = [k for k in PM10_LABELS if k in set(s[col].astype(str))]
        sns.boxplot(data=s, x=col, y="usage_ratio", order=order, color="#7a8b99", ax=ax, fliersize=2)
        c = s[col].astype(str).value_counts()
        for i, k in enumerate(order):
            ax.text(i, ax.get_ylim()[1] * 0.97, f"n={c.get(k, 0)}", ha="center", va="top", fontsize=9)
        ax.set_xlabel(lab)
        ax.set_ylabel("상대 이용량")
    fig.suptitle("대기오염 등급별 상대 이용량 (계절 효과가 섞여 있으므로 회귀 결과와 함께 해석)")
    save(fig, out, "04_air_grade.png")

    # 05 상관 히트맵
    cols = {"usage_ratio": "상대이용량", "tavg": "평균기온", "tmin": "최저기온", "tmax": "최고기온",
            "rain": "강수량", "pm10": "PM10", "pm25": "PM2.5"}
    cm = d[list(cols)].rename(columns=cols).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman 상관계수")
    save(fig, out, "05_corr.png")

    # 06 구 × 기온 구간 히트맵
    pv = d.dropna(subset=["tavg"]).pivot_table(index="gu", columns="temp_bin", values="usage_ratio",
                                            aggfunc=["mean", "count"], observed=True)
    mean, cnt = pv["mean"], pv["count"]
    mean = mean.where(cnt >= MIN_N)
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * mean.shape[1] + 3), 9))
    sns.heatmap(mean, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "상대 이용량"})
    ax.set_xlabel("평균기온 구간(℃)")
    ax.set_ylabel("")
    ax.set_title(f"자치구 × 기온 구간 상대 이용량 (셀당 {MIN_N}일 미만은 공란)")
    save(fig, out, "06_gu_temp_heatmap.png")

    # 07 구별 민감도
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    for ax, col, ttl, color in [(axes[0], "기온 1℃당 변화(%)", "기온 1℃ 상승 시 이용량 변화(%)", "#d1495b"),
                                (axes[1], "강수일 이용량 변화(%)", "강수일 vs 무강수일 이용량 변화(%)", "#4a90c2")]:
        s = sens.dropna(subset=[col]).sort_values(col)
        ax.barh(s["자치구"], s[col], color=color, alpha=.85)
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(ttl)
    save(fig, out, "07_gu_sensitivity.png")


# ----------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usage", default="data/usage/**/*_merge.csv")
    p.add_argument("--rain", default="data/rain/일일강수량_*.csv")
    p.add_argument("--temp", default="data/temp/일일기온_*.csv")
    p.add_argument("--air", default="data/air/일별평균대기오염도_*.csv")
    p.add_argument("--out", default="result")
    p.add_argument("--holiday-file", default=None,
                    help="추가 공휴일 CSV(컬럼명 date, 예: 2025-01-27). holidays 패키지에 빠진 임시공휴일 보정용")
    p.add_argument("--keep-suspect", action="store_true",
                    help="이용량 급감일(의심일)을 제외하지 않고 분석에 포함")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_font()

    df = add_features(build_table(args), keep_suspect=args.keep_suspect,
                        holiday_file=args.holiday_file, out=args.out)
    print(f"[2] 통합 테이블: {len(df):,}행, 기간 {df['date'].min().date()} ~ {df['date'].max().date()}, 구 {df['gu'].nunique()}개")

    sus = sorted(df.loc[df["suspect"], "date"].dt.date.unique())
    if sus:
        print(f"  [주의] 이용량이 같은 유형 중앙값의 {SUSPECT_RATIO:.0%} 미만인 날: {[str(s) for s in sus]} "
            f"→ {'분석에 포함' if args.keep_suspect else '분석에서 제외'}. 폭우·폭설 등 실제 날씨 영향인지, 기록 누락인지 반드시 확인하세요.")
    q = df.groupby("gu").agg(일수=("date", "count"), 기온결측=("tavg", lambda s: s.isna().sum()),
                            강수결측=("rain", lambda s: s.isna().sum()), 미세먼지결측=("pm10", lambda s: s.isna().sum()))
    q.to_csv(os.path.join(args.out, "data_quality_by_gu.csv"), encoding="utf-8-sig")
    df.to_csv(os.path.join(args.out, "daily_gu_weather.csv"), index=False, encoding="utf-8-sig")

    print("[3] 회귀: 로그 이용량 ~ 기온 + 강수 + 미세먼지 + 주말·공휴일 + 구(+월) 고정효과")
    reg = regression(df)
    reg.to_csv(os.path.join(args.out, "regression_summary.csv"), index=False, encoding="utf-8-sig")
    print(reg.to_string(index=False), f"\n  (관측 {reg.attrs['n']:,}행, 오차는 날짜 단위 군집 보정)")

    sens = gu_sensitivity(df)
    sens.to_csv(os.path.join(args.out, "gu_sensitivity.csv"), index=False, encoding="utf-8-sig")

    print("[4] 그래프 저장")
    plot_all(df, sens, args.out)
    print(f"완료 → {args.out}/")


if __name__ == "__main__":
    main()