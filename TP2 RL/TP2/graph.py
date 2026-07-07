import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

def draw_node(ax, x, y, label, radius=0.33, facecolor="#f2d27c", edgecolor="black", fontsize=10):
    c = Circle((x, y), radius=radius, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(c)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)

def draw_arrow(ax, x1, y1, x2, y2, color="black", lw=1.2, rad=0.0, alpha=1.0, arrowstyle="-|>"):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=arrowstyle,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        alpha=alpha,
    )
    ax.add_patch(arr)

def draw_group_box(ax, x, y, w, h, title):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.2",
        linewidth=1.0,
        edgecolor="black",
        facecolor="white",
        alpha=0.92
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h + 0.25, title, ha="center", va="bottom", fontsize=11)

def pos_to_xy_2x2(pos: int):
    """
    Mapping consistent with the statement 'Pos = 0 bottom-left' and row-major indexing:
      (1,1)->0, (2,1)->1, (1,2)->2, (2,2)->3
    Coordinates are 1-based (as in board0.txt).
    """
    mapping = {
        0: (1, 1),
        1: (2, 1),
        2: (1, 2),
        3: (2, 2),
    }
    return mapping[pos]

def generate_markov_board0(out_path="markov_board0.png"):
    # board0.txt content:
    # 2
    # 1 1 H
    # 2 2 F
    N = 2
    house_xy = (1, 1)
    food_xy = (2, 2)

    # Convert to Pos indices under the assumed mapping
    pos_house = 0  # (1,1)
    pos_food = 3   # (2,2)

    # Layout
    left_x, right_x = 2.0, 6.4
    # Place nodes roughly in a square arrangement (top to bottom)
    # We'll list positions in this order to avoid a vertical "stack" look:
    # 2 (top-left), 3 (top-right), 0 (bottom-left), 1 (bottom-right)
    layout_order = [2, 3, 0, 1]
    y_coords = [6.0, 6.0, 2.2, 2.2]
    x_offsets = [-0.7, 0.7, -0.7, 0.7]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Group boxes
    box_w, box_h = 3.0, 5.6
    left_box = (left_x - 1.5, 1.0, box_w, box_h)
    right_box = (right_x - 1.5, 1.0, box_w, box_h)
    draw_group_box(ax, *left_box, title="Food state f = 0 (not eaten)")
    draw_group_box(ax, *right_box, title="Food state f = 1 (eaten)")

    # Build node coordinates for each layer
    left_nodes = {}
    right_nodes = {}

    for pos, y, dx in zip(layout_order, y_coords, x_offsets):
        xL = left_x + dx
        xR = right_x + dx
        left_nodes[pos] = (xL, y)
        right_nodes[pos] = (xR, y)

        xy = pos_to_xy_2x2(pos)
        tag = ""
        if xy == house_xy:
            tag = " (H)"
        if xy == food_xy:
            tag = " (F)"

        # Label includes Pos and coordinates to make it unique and rigorous
        label = f"Pos {pos}\n({xy[0]},{xy[1]}){tag}"
        draw_node(ax, xL, y, label, fontsize=9)
        draw_node(ax, xR, y, label, fontsize=9)

    # Exact 2x2 move graph (valid moves only)
    # Edges: 0<->1, 0<->2, 1<->3, 2<->3
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]

    def connect_bidirectional(layer_nodes, rad=0.15, color="black"):
        for a, b in edges:
            x1, y1 = layer_nodes[a]
            x2, y2 = layer_nodes[b]
            # two opposite curved arrows to show bidirectional transitions
            draw_arrow(ax, x1, y1, x2, y2, color=color, rad=rad)
            draw_arrow(ax, x2, y2, x1, y1, color=color, rad=-rad)

    connect_bidirectional(left_nodes, rad=0.12, color="black")
    connect_bidirectional(right_nodes, rad=0.12, color="black")

    # Cross-layer transition: eating food at Pos 3 (2,2)
    fx, fy = left_nodes[pos_food]
    tx, ty = right_nodes[pos_food]
    draw_arrow(ax, fx + 0.35, fy, tx - 0.35, ty, color="red", lw=2.2, rad=0.0)
    ax.text((fx + tx) / 2, fy + 0.45, "eat food: f=0 → f=1", ha="center", va="bottom", fontsize=10, color="red")

    # Title and caption
    ax.text(0.25, 7.25, "Figure — Markov chain representation for board0.txt (2×2, H at (1,1), F at (2,2))",
            fontsize=12, fontweight="bold", ha="left", va="top")
    ax.text(0.25, 0.35,
            "State space split by food status. Within-layer arrows represent valid moves; red arrow is the irreversible food consumption transition.",
            fontsize=9, ha="left", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    generate_markov_board0("markov_board0.png")
