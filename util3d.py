import matplotlib.pyplot as plt

name_legend = {
    "ac": "brown_cricket",
    "bc": "black_cricket",
    "bf": "blow_fly",
    "bl": "buffalo_bettle_larva",
    "bp": "blow_fly_pupa",
    "cf": "curly-wing_fly",
    "gh": "grasshopper",
    "ma": "maggot",
    "ml": "mealworm",
    "pp": "green_bottle_fly_pupa",
    "sl": "soldier_fly_larva",
    "wo": "woodlice",
}

def show_volume(volume, label=None, size=1, **kwargs):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(size * 6, size * 3), tight_layout=True)
    if label is not None:
        fig.suptitle(list(name_legend.values())[label].replace("_", " "))
    ax0.imshow(volume.max(0), **kwargs)
    ax1.imshow(volume.max(1), **kwargs)
    ax2.imshow(volume.max(2), **kwargs)
    plt.show()
