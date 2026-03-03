"""
generate_mockups.py
-------------------
Generates four mockup PNGs for the COMP 6934 project proposal.
All data is synthetic but structurally faithful to the OWID CO2 dataset.

Output files:
    viz1_bubble_map.png   - Proportional Symbol Map
    viz2_linechart.png    - Small-Multiple Line Charts
    viz3_scatter.png      - Annotated Log-Log Scatterplot
    viz4_decoupling.png   - Connected Scatterplot
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

RNG = np.random.default_rng(42)

FUEL_COLORS = {
    "Coal":    "#2c2c54",
    "Oil":     "#c0392b",
    "Gas":     "#e67e22",
    "Cement":  "#95a5a6",
    "Flaring": "#f1c40f",
}
LUC_POS_COLOR = "#27ae60"
LUC_NEG_COLOR = "#1abc9c"

INCOME_COLORS = {
    "Low income":          "#fef0d9",
    "Lower-middle income": "#fdcc8a",
    "Upper-middle income": "#fc8d59",
    "High income":         "#b30000",
}

YEARS_1950 = np.arange(1950, 2023)


# ============================================================
# VIZ 1 - Proportional Symbol Map
# ============================================================
def make_viz1():
    countries = {
        "USA":          dict(lon=-100, lat=38,  cumco2=410,  perco2=14.2),
        "China":        dict(lon=104,  lat=35,  cumco2=230,  perco2=7.4),
        "Russia":       dict(lon=90,   lat=60,  cumco2=115,  perco2=11.5),
        "Germany":      dict(lon=10,   lat=51,  cumco2=92,   perco2=7.8),
        "UK":           dict(lon=-2,   lat=54,  cumco2=82,   perco2=4.9),
        "Japan":        dict(lon=138,  lat=37,  cumco2=76,   perco2=8.1),
        "India":        dict(lon=78,   lat=22,  cumco2=52,   perco2=1.9),
        "Canada":       dict(lon=-96,  lat=56,  cumco2=46,   perco2=14.8),
        "Australia":    dict(lon=134,  lat=-25, cumco2=36,   perco2=14.0),
        "France":       dict(lon=2,    lat=47,  cumco2=33,   perco2=4.2),
        "Brazil":       dict(lon=-52,  lat=-10, cumco2=29,   perco2=2.4),
        "S. Korea":     dict(lon=128,  lat=37,  cumco2=23,   perco2=11.2),
        "Mexico":       dict(lon=-102, lat=24,  cumco2=18,   perco2=3.5),
        "Indonesia":    dict(lon=118,  lat=-2,  cumco2=14,   perco2=2.0),
        "S. Africa":    dict(lon=25,   lat=-29, cumco2=13,   perco2=6.8),
        "Iran":         dict(lon=53,   lat=33,  cumco2=11,   perco2=7.6),
        "Saudi Arabia": dict(lon=45,   lat=24,  cumco2=10,   perco2=14.4),
        "Poland":       dict(lon=20,   lat=52,  cumco2=9,    perco2=8.1),
        "Nigeria":      dict(lon=8,    lat=9,   cumco2=5,    perco2=0.7),
        "Argentina":    dict(lon=-64,  lat=-34, cumco2=7,    perco2=4.0),
        "Turkey":       dict(lon=35,   lat=39,  cumco2=8,    perco2=4.5),
        "Egypt":        dict(lon=30,   lat=26,  cumco2=4,    perco2=2.4),
        "Pakistan":     dict(lon=69,   lat=30,  cumco2=3,    perco2=0.9),
        "Thailand":     dict(lon=101,  lat=15,  cumco2=6,    perco2=3.8),
    }
    df = pd.DataFrame(countries).T.reset_index().rename(
        columns={"index": "country"})
    df = df.astype({"lon": float, "lat": float,
                   "cumco2": float, "perco2": float})

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_facecolor("#d6eaf8")

    land = [
        mpatches.FancyBboxPatch((-168, 15),  110, 60,
                                color="#c8d6a0", zorder=0),
        mpatches.FancyBboxPatch((-85,  -58),  65, 42,
                                color="#c8d6a0", zorder=0),
        mpatches.FancyBboxPatch((-25,  35),   55, 35,
                                color="#c8d6a0", zorder=0),
        mpatches.FancyBboxPatch((25,   -38),  55, 80,
                                color="#c8d6a0", zorder=0),
        mpatches.FancyBboxPatch((60,   -10), 105, 75,
                                color="#c8d6a0", zorder=0),
        mpatches.FancyBboxPatch((112,  -45),  50, 40,
                                color="#c8d6a0", zorder=0),
    ]
    for p in land:
        ax.add_patch(p)

    cmap = LinearSegmentedColormap.from_list("perco2", ["#fff7bc", "#d95f0e"])
    norm = Normalize(vmin=0, vmax=16)
    sizes = (df["cumco2"] / df["cumco2"].max()) * 2400 + 60

    sc = ax.scatter(
        df["lon"], df["lat"],
        s=sizes, c=df["perco2"], cmap=cmap, norm=norm,
        alpha=0.85, edgecolors="white", linewidths=0.7, zorder=3
    )

    for _, row in df.iterrows():
        if row["cumco2"] >= 25 or row["country"] in ["India", "Nigeria", "S. Africa"]:
            ax.annotate(
                row["country"], (row["lon"], row["lat"]),
                xytext=(0, 7), textcoords="offset points",
                fontsize=7, ha="center", color="#1a252f", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")]
            )

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.02, shrink=0.7)
    cbar.set_label(
        "CO\u2082 per capita \u2014 2022  (t / person)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    for cum, label in [(50, "50 Gt"), (150, "150 Gt"), (400, "400 Gt")]:
        s = (cum / df["cumco2"].max()) * 2400 + 60
        ax.scatter([], [], s=s, c="#aaa", alpha=0.75,
                   edgecolors="white", linewidths=0.7, label=label)
    ax.legend(title="Cumulative CO\u2082", title_fontsize=8.5, fontsize=8,
              loc="lower left", framealpha=0.9, scatterpoints=1, labelspacing=1.0)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 85)
    ax.set_xlabel("Longitude", fontsize=8.5)
    ax.set_ylabel("Latitude", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.set_title(
        "Visualization 1 \u2014 Proportional Symbol Map  (Year: 2022)\n"
        "Bubble area \u2192 Cumulative CO\u2082 (Gt)     "
        "Color \u2192 CO\u2082 per capita (t/person)",
        fontsize=10
    )

    plt.savefig("/mnt/user-data/outputs/viz1_bubble_map.png")
    plt.close()
    print("  viz1_bubble_map.png")


# ============================================================
# VIZ 2 - Small-Multiple Line Charts
# ============================================================
def make_viz2():
    fuels = list(FUEL_COLORS.keys())

    def gen_country(seed, coal_peak, oil_peak, gas_rise, scale=1.0):
        rng = np.random.default_rng(seed)
        n = len(YEARS_1950)
        t = np.linspace(0, 1, n)
        coal = scale * coal_peak * np.exp(-((t - 0.45)**2) / 0.10) \
            * (1 + 0.04 * rng.standard_normal(n))
        oil = scale * oil_peak * (1 / (1 + np.exp(-10 * (t - 0.35)))) \
            * np.exp(-1.8 * (t - 0.72)**2) * (1 + 0.04 * rng.standard_normal(n))
        gas = scale * gas_rise * (1 / (1 + np.exp(-9 * (t - 0.55)))) \
            * (1 + 0.05 * rng.standard_normal(n))
        cem = scale * 0.10 * t**1.8 * (1 + 0.07 * rng.standard_normal(n))
        flar = scale * 0.05 * np.ones(n) * (1 + 0.12 * rng.standard_normal(n))
        luc = scale * 0.6 * np.exp(-((t - 0.60)**2) / 0.05) \
            * (1 + 0.10 * rng.standard_normal(n))
        luc[int(0.8*n):] -= 0.35 * scale
        return pd.DataFrame({
            "year":    YEARS_1950,
            "Coal":    np.clip(coal,  0, None),
            "Oil":     np.clip(oil,   0, None),
            "Gas":     np.clip(gas,   0, None),
            "Cement":  np.clip(cem,   0, None),
            "Flaring": np.clip(flar,  0, None),
            "LUC":     luc,
        })

    country_params = {
        "USA":     dict(seed=1, coal_peak=2.5, oil_peak=2.2, gas_rise=2.0, scale=1.0),
        "China":   dict(seed=2, coal_peak=7.0, oil_peak=0.8, gas_rise=0.6, scale=1.5),
        "Russia":  dict(seed=3, coal_peak=1.2, oil_peak=1.0, gas_rise=1.4, scale=0.8),
        "India":   dict(seed=4, coal_peak=1.6, oil_peak=0.5, gas_rise=0.4, scale=1.1),
        "Germany": dict(seed=5, coal_peak=0.9, oil_peak=0.5, gas_rise=0.4, scale=0.6),
        "UK":      dict(seed=6, coal_peak=0.7, oil_peak=0.4, gas_rise=0.3, scale=0.5),
        "Japan":   dict(seed=7, coal_peak=0.5, oil_peak=0.9, gas_rise=0.5, scale=0.6),
        "Brazil":  dict(seed=8, coal_peak=0.2, oil_peak=0.4, gas_rise=0.3, scale=0.4),
    }
    countries = list(country_params.keys())
    data = {c: gen_country(**country_params[c]) for c in countries}

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5),
                             sharex=True, sharey=False)
    axes = axes.flatten()

    for i, country in enumerate(countries):
        ax = axes[i]
        df = data[country]

        for fuel in fuels:
            ax.plot(df["year"], df[fuel],
                    color=FUEL_COLORS[fuel], lw=1.6, label=fuel)

        # LUC as filled area on twin axis (diverging)
        ax2 = ax.twinx()
        ax2.fill_between(df["year"], 0, df["LUC"],
                         where=df["LUC"] >= 0,
                         color=LUC_POS_COLOR, alpha=0.25)
        ax2.fill_between(df["year"], 0, df["LUC"],
                         where=df["LUC"] < 0,
                         color=LUC_NEG_COLOR, alpha=0.30)
        ax2.axhline(0, color=LUC_POS_COLOR, lw=0.6, ls="--")
        ax2.set_ylim(-1, 2)
        ax2.tick_params(right=False, labelright=False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        for yr, lbl in [(1997, "K"), (2016, "P")]:
            ax.axvline(yr, color="#bdc3c7", lw=0.8, ls=":")
            ax.text(yr + 0.5, ax.get_ylim()[1] * 0.92,
                    lbl, fontsize=6.5, color="#7f8c8d")

        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.set_xlim(1950, 2022)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7.5)
        ax.set_ylabel("MtCO\u2082" if i % 4 == 0 else "", fontsize=8)
        ax.set_xlabel("Year" if i >= 4 else "", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fuel_handles = [mlines.Line2D([], [], color=FUEL_COLORS[f], lw=2, label=f)
                    for f in fuels]
    luc_handles = [
        mpatches.Patch(color=LUC_POS_COLOR, alpha=0.5,
                       label="Land-use chg (+)"),
        mpatches.Patch(color=LUC_NEG_COLOR, alpha=0.5,
                       label="Land-use chg (\u2212)"),
    ]
    fig.legend(
        handles=fuel_handles + luc_handles,
        loc="lower center", ncol=7,
        fontsize=8.5, title="Fuel source", title_fontsize=9,
        bbox_to_anchor=(0.5, -0.04), framealpha=0.9
    )
    fig.suptitle(
        "Visualization 2 \u2014 Small-Multiple Line Charts\n"
        "Fuel-source CO\u2082 composition by country, 1950\u20132022"
        "  |  K = Kyoto 1997   P = Paris 2016",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    plt.savefig("/mnt/user-data/outputs/viz2_linechart.png")
    plt.close()
    print("  viz2_linechart.png")


# ============================================================
# VIZ 3 - Annotated Log-Log Scatterplot
# ============================================================
def make_viz3():
    income_order = ["Low income", "Lower-middle income",
                    "Upper-middle income", "High income"]
    records = [
        ("USA",           14.5, 17.2, 330,  "High income"),
        ("UK",             4.8,  7.8,  67,  "High income"),
        ("Germany",        7.9,  9.6,  83,  "High income"),
        ("Australia",     14.1, 15.3,  26,  "High income"),
        ("Japan",          8.2,  8.9, 126,  "High income"),
        ("S. Korea",      11.2, 12.1,  52,  "High income"),
        ("France",         4.1,  6.8,  67,  "High income"),
        ("Canada",        14.9, 15.6,  38,  "High income"),
        ("Norway",         7.8,  9.2,   5,  "High income"),
        ("Sweden",         3.5,  6.4,  10,  "High income"),
        ("Switzerland",    3.8,  7.1,   9,  "High income"),
        ("Singapore",      9.2, 18.6,   6,  "High income"),
        ("Hong Kong",      3.2, 12.1,   7,  "High income"),
        ("Poland",         7.8,  7.2,  38,  "High income"),
        ("Saudi Arabia",  14.4, 10.2,  35,  "High income"),
        ("UAE",           20.1, 14.8,  10,  "High income"),
        ("Qatar",         31.5, 18.2,   3,  "High income"),
        ("China",          7.4,  6.5, 1400,  "Upper-middle income"),
        ("Russia",        11.6,  9.8, 145,  "Upper-middle income"),
        ("Brazil",         2.4,  2.2, 215,  "Upper-middle income"),
        ("Mexico",         3.5,  3.3, 130,  "Upper-middle income"),
        ("S. Africa",      6.9,  5.8,  60,  "Upper-middle income"),
        ("Iran",           7.6,  6.9,  85,  "Upper-middle income"),
        ("Turkey",         4.5,  4.3,  84,  "Upper-middle income"),
        ("Kazakhstan",    13.5, 11.2,  19,  "Upper-middle income"),
        ("India",          1.9,  1.7, 1380,  "Lower-middle income"),
        ("Indonesia",      2.0,  1.9, 273,  "Lower-middle income"),
        ("Vietnam",        2.5,  2.3,  98,  "Lower-middle income"),
        ("Philippines",    1.2,  1.3, 111,  "Lower-middle income"),
        ("Nigeria",        0.7,  0.8, 213,  "Low income"),
        ("Ethiopia",       0.1,  0.2, 117,  "Low income"),
        ("DRC",            0.05, 0.1, 100,  "Low income"),
        ("Mozambique",     0.12, 0.15, 32,  "Low income"),
    ]
    df = pd.DataFrame(records,
                      columns=["country", "prod_pc", "cons_pc", "pop_M", "income"])
    df["income"] = pd.Categorical(df["income"],
                                  categories=income_order, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f8f9fa")

    lim_max = 38
    ax.plot([0.04, lim_max], [0.04, lim_max],
            color="#7f8c8d", lw=1.3, ls="--", zorder=1)
    ax.text(22, 17, "Production = Consumption\n(neutrality)",
            fontsize=7.5, color="#7f8c8d", rotation=38, ha="center")

    ax.fill_between([0.04, lim_max], [0.04, lim_max], [lim_max, lim_max],
                    alpha=0.04, color="#e74c3c", zorder=0)
    ax.fill_between([0.04, lim_max], [0.04, 0.04],   [0.04, lim_max],
                    alpha=0.04, color="#2980b9", zorder=0)

    for income in income_order:
        grp = df[df["income"] == income]
        sizes = (grp["pop_M"] / df["pop_M"].max()) * 750 + 28
        ax.scatter(grp["prod_pc"], grp["cons_pc"],
                   s=sizes, c=INCOME_COLORS[income],
                   edgecolors="#2c3e50", linewidths=0.55,
                   alpha=0.90, zorder=3, label=income)

    annotate = {
        "USA":          (0.6, -0.6),
        "UK":           (0.3,  0.4),
        "Germany":      (0.3,  0.4),
        "China":        (-1.8, -0.6),
        "India":        (0.3, -0.5),
        "Singapore":    (0.3,  0.4),
        "Hong Kong":    (0.3,  0.4),
        "Qatar":        (0.3,  0.3),
        "Russia":       (0.3, -0.5),
        "Saudi Arabia": (-2.5,  0.2),
        "S. Korea":     (0.3,  0.3),
        "Nigeria":      (0.25, 0.15),
    }
    for _, row in df.iterrows():
        if row["country"] in annotate:
            dx, dy = annotate[row["country"]]
            ax.annotate(
                row["country"],
                (row["prod_pc"], row["cons_pc"]),
                xytext=(row["prod_pc"] + dx, row["cons_pc"] + dy),
                fontsize=7.5, color="#1a252f", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#bdc3c7", lw=0.6),
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
            )

    ax.text(0.055, 20,
            "Net emissions IMPORTERS\n(consume > produce)",
            fontsize=8, color="#c0392b", alpha=0.75, style="italic")
    ax.text(9, 0.055,
            "Net emissions EXPORTERS\n(produce > consume)",
            fontsize=8, color="#2980b9", alpha=0.75, style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.04, lim_max)
    ax.set_ylim(0.04, lim_max)
    ax.set_xlabel(
        "CO\u2082 per capita \u2014 Production-based  (t/person, log scale)", fontsize=9)
    ax.set_ylabel(
        "CO\u2082 per capita \u2014 Consumption-based  (t/person, log scale)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.45)

    for pop, lbl in [(50, "50 M"), (400, "400 M"), (1400, "1.4 B")]:
        s = (pop / df["pop_M"].max()) * 750 + 28
        ax.scatter([], [], s=s, c="#bdc3c7", edgecolors="#2c3e50",
                   linewidths=0.5, alpha=0.85, label=lbl)

    ax.legend(fontsize=8, loc="lower right", framealpha=0.92,
              title="Income group / Population", title_fontsize=8.5)
    ax.set_title(
        "Visualization 3 \u2014 Annotated Log-Log Scatterplot  (Year: 2019)\n"
        "Production vs. Consumption CO\u2082 per capita by income group",
        fontsize=10
    )

    plt.savefig("/mnt/user-data/outputs/viz3_scatter.png")
    plt.close()
    print("  viz3_scatter.png")


# ============================================================
# VIZ 4 - Connected Scatterplot
# ============================================================
def make_viz4():
    yrs = np.arange(1990, 2023)
    n = len(yrs)
    t = np.linspace(0, 1, n)
    rng = np.random.default_rng(77)

    # Germany: GDP per capita rises steadily, CO2 per capita falls
    gdp_pc = 28000 + 12000 * t + 800 * rng.standard_normal(n)
    gdp_pc[13:16] -= 2200   # early-2000s slowdown
    gdp_pc[18:20] -= 4500   # 2008-09 GFC dip
    gdp_pc = np.maximum.accumulate(gdp_pc) - 800

    co2_pc = 11.5 * np.exp(-0.028 * np.arange(n)) + \
        0.18 * rng.standard_normal(n)
    co2_pc[18:20] -= 0.6
    co2_pc = np.clip(co2_pc, 5, None)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#f8f9fa")

    cmap_yr = plt.cm.plasma
    norm_yr = Normalize(vmin=yrs[0], vmax=yrs[-1])

    points = np.array([gdp_pc, co2_pc]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap_yr, norm=norm_yr, lw=2.4, zorder=3)
    lc.set_array(yrs[:-1].astype(float))
    ax.add_collection(lc)

    sc = ax.scatter(gdp_pc, co2_pc,
                    c=yrs, cmap=cmap_yr, norm=norm_yr,
                    s=32, zorder=4, edgecolors="white", linewidths=0.5)

    label_yrs = set(range(1990, 2023, 4)) | {2022}
    for i, yr in enumerate(yrs):
        if yr in label_yrs:
            ax.annotate(str(yr), (gdp_pc[i], co2_pc[i]),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, color="#2c3e50")

    for i in [4, 14, 25]:
        ax.annotate("",
                    xy=(gdp_pc[i+1], co2_pc[i+1]),
                    xytext=(gdp_pc[i],   co2_pc[i]),
                    arrowprops=dict(arrowstyle="-|>", color="#7f8c8d",
                                    lw=1.0, mutation_scale=12))

    # Decoupling threshold
    ax.axhline(co2_pc[0], color="#e74c3c", lw=1.0, ls="--", alpha=0.6)
    ax.text(gdp_pc.min() + 200, co2_pc[0] + 0.1,
            "CO\u2082 level in 1990  (absolute decoupling threshold)",
            fontsize=7.5, color="#e74c3c", alpha=0.85)

    ax.annotate(
        "\u2192 rising GDP per capita\n\u2193 falling CO\u2082 per capita\n\u2198 = decoupling",
        xy=(0.03, 0.97), xycoords="axes fraction",
        fontsize=8, va="top", color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec="#bdc3c7", lw=0.9, alpha=0.95)
    )

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02, shrink=0.75)
    cbar.set_label("Year", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.autoscale()
    ax.set_xlabel("GDP per capita  (USD, 2015 constant)", fontsize=9)
    ax.set_ylabel("CO\u2082 per capita  (t / person)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(ls=":", lw=0.4, alpha=0.45)
    ax.set_title(
        "Visualization 4 \u2014 Connected Scatterplot  |  Country: Germany  |  1990\u20132022\n"
        "Trajectory through GDP per capita \u00d7 CO\u2082 per capita space",
        fontsize=10
    )

    plt.savefig("/mnt/user-data/outputs/viz4_decoupling.png")
    plt.close()
    print("  viz4_decoupling.png")


# ============================================================
if __name__ == "__main__":
    print("Generating mockup PNGs ...\n")
    make_viz1()
    make_viz2()
    make_viz3()
    make_viz4()
    print("\nDone. Files written to /mnt/user-data/outputs/")
