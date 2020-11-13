import numpy as np
from stringcolor import cs

_ = [
    "#4682B4",
    "#87CEEB",
    "#87CEFA",
    "#00BFFF",
    "#1E90FF",
    "#6495ED",
    "#4169E1",
    "#0000FF",
    "#0000CD",
    "#00008B",
    "#000080"
]

colors = [
    "#f5f5ff",
    "#ffe0e0",
    "#ffb6b6",
    "#ff8d8d",
    "#ff6363",
    "#ff3939",
    "#ff1010",
    "#f20000",
    "#dd0000",
    "#c80000",
    "#b40000",
    "#9f0000",
    "#8a0000",
]

def print_color_string(string, ws):
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    for s, w in zip(string, ws):
        i = int(w * len(colors))
        print(cs(s, colors[i]), end="")

if __name__ == "__main__":
    # for testing
    print_color_string("abcdef", [1, 2, 3, 4, 5])
