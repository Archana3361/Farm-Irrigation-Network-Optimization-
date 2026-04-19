"""
============================================================
  Farm Irrigation Network Optimization
  ─────────────────────────────────────
  Algorithms: Kruskal's MST · Dijkstra's Shortest Path · 0/1 Knapsack
  GUI: Tkinter  |  Visualization: NetworkX + Matplotlib
============================================================
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random
import heapq
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────
#  COLOUR PALETTE  (agricultural / earthy theme)
# ──────────────────────────────────────────────────────────
BG_DARK    = "#1a2e1a"   # very dark green – window background
BG_PANEL   = "#243324"   # slightly lighter – side panel
BG_CARD    = "#2d422d"   # card / button area
ACCENT_GRN = "#5aad5a"   # bright green – primary buttons
ACCENT_YLW = "#e8c84a"   # golden yellow – MST highlights
ACCENT_CYN = "#4acce8"   # cyan – Dijkstra path
ACCENT_RED = "#e8614a"   # warm red – Knapsack
TEXT_MAIN  = "#e8f5e8"   # near-white text
TEXT_DIM   = "#8aab8a"   # dimmed label text
FONT_MAIN  = ("Consolas", 10)
FONT_BOLD  = ("Consolas", 10, "bold")
FONT_TITLE = ("Consolas", 13, "bold")
FONT_HEAD  = ("Consolas", 11, "bold")


# ──────────────────────────────────────────────────────────
#  ALGORITHM 1 – KRUSKAL'S MST
#  Uses Union-Find (Disjoint Set Union) for cycle detection.
# ──────────────────────────────────────────────────────────
class UnionFind:
    """Simple Union-Find (DSU) data structure."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False   # already in the same set → cycle
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def kruskal(nodes, edges):
    """
    Kruskal's Algorithm – Minimum Spanning Tree.
    Parameters
    ----------
    nodes : list of node labels
    edges : list of (weight, u, v) tuples
    Returns
    -------
    mst_edges : list of (weight, u, v) in the MST
    total_cost : int / float
    """
    node_index = {n: i for i, n in enumerate(nodes)}
    sorted_edges = sorted(edges, key=lambda e: e[0])   # sort by weight
    uf = UnionFind(len(nodes))
    mst_edges, total_cost = [], 0

    for weight, u, v in sorted_edges:
        if uf.union(node_index[u], node_index[v]):
            mst_edges.append((weight, u, v))
            total_cost += weight
            if len(mst_edges) == len(nodes) - 1:
                break   # MST complete

    return mst_edges, total_cost


# ──────────────────────────────────────────────────────────
#  ALGORITHM 2 – DIJKSTRA'S SHORTEST PATH
#  Min-heap based priority queue implementation.
# ──────────────────────────────────────────────────────────
def dijkstra(graph, source):
    """
    Dijkstra's Algorithm – Single-source shortest paths.
    Parameters
    ----------
    graph  : dict  { node: [(weight, neighbour), ...] }
    source : starting node
    Returns
    -------
    dist : dict  { node: shortest distance from source }
    prev : dict  { node: predecessor on shortest path }
    """
    dist = {n: float('inf') for n in graph}
    prev = {n: None for n in graph}
    dist[source] = 0
    heap = [(0, source)]   # (distance, node)

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue   # stale entry
        for weight, v in graph[u]:
            alt = dist[u] + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    return dist, prev


def reconstruct_path(prev, source, target):
    """Walk the predecessor map to reconstruct a path."""
    path, node = [], target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    if path and path[0] == source:
        return path
    return []   # no path exists


# ──────────────────────────────────────────────────────────
#  ALGORITHM 3 – 0/1 KNAPSACK (Water Distribution)
#  Classic bottom-up DP approach.
# ──────────────────────────────────────────────────────────
def knapsack_01(capacity, weights, values, names):
    """
    0/1 Knapsack Algorithm – Optimal crop-water selection.
    Parameters
    ----------
    capacity : total water units available
    weights  : list of water requirement per crop
    values   : list of yield/profit per crop
    names    : list of crop names
    Returns
    -------
    max_value    : optimal total value
    chosen_items : list of selected crop names
    chosen_detail: list of (name, weight, value) tuples
    """
    n = len(weights)
    # Build DP table  (n+1) × (capacity+1)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i - 1][w]
            # Take item i-1 if it fits
            if weights[i - 1] <= w:
                take = dp[i - 1][w - weights[i - 1]] + values[i - 1]
                if take > dp[i][w]:
                    dp[i][w] = take

    # Backtrack to find chosen items
    chosen_items, chosen_detail = [], []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen_items.append(names[i - 1])
            chosen_detail.append((names[i - 1], weights[i - 1], values[i - 1]))
            w -= weights[i - 1]

    return dp[n][capacity], chosen_items, chosen_detail


# ──────────────────────────────────────────────────────────
#  RANDOM GRAPH GENERATOR
# ──────────────────────────────────────────────────────────
def generate_random_graph(num_nodes=None):
    """
    Generate a random connected weighted graph representing the farm.
    Returns
    -------
    nodes : list of node labels  e.g. ['F1', 'F2', ...]
    edges : list of (weight, u, v) – pipe connections
    adj   : adjacency list for Dijkstra
    """
    if num_nodes is None:
        num_nodes = random.randint(5, 8)

    nodes = [f"F{i+1}" for i in range(num_nodes)]
    edges = set()

    # Guarantee connectivity: chain all nodes first
    shuffled = nodes[:]
    random.shuffle(shuffled)
    for i in range(len(shuffled) - 1):
        w = random.randint(1, 20)
        u, v = shuffled[i], shuffled[i + 1]
        if u > v:
            u, v = v, u
        edges.add((w, u, v))

    # Add a few extra random edges for variety
    extra = random.randint(2, num_nodes)
    for _ in range(extra):
        u, v = random.sample(nodes, 2)
        if u > v:
            u, v = v, u
        if not any(e[1] == u and e[2] == v for e in edges):
            w = random.randint(1, 20)
            edges.add((w, u, v))

    edges = list(edges)

    # Build adjacency list
    adj = {n: [] for n in nodes}
    for w, u, v in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))

    return nodes, edges, adj


# ──────────────────────────────────────────────────────────
#  CROP DATA GENERATOR (for Knapsack)
# ──────────────────────────────────────────────────────────
CROP_POOL = [
    ("Wheat",   3, 4),   # (name, water_units, yield_value)
    ("Rice",    5, 8),
    ("Corn",    4, 6),
    ("Cotton",  6, 9),
    ("Tomato",  3, 5),
    ("Potato",  2, 3),
    ("Sugarcane", 7, 11),
    ("Soybean", 4, 7),
    ("Barley",  2, 4),
    ("Millet",  1, 2),
]

def generate_crops(num_nodes):
    """Select a random subset of crops matching the node count."""
    k = min(num_nodes, len(CROP_POOL))
    selected = random.sample(CROP_POOL, k)
    names   = [c[0] for c in selected]
    weights = [c[1] for c in selected]
    values  = [c[2] for c in selected]
    return names, weights, values


# ──────────────────────────────────────────────────────────
#  MAIN APPLICATION CLASS
# ──────────────────────────────────────────────────────────
class FarmIrrigationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Farm Irrigation Network Optimization")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1280x780")
        self.root.resizable(True, True)

        # State
        self.nodes      = []
        self.edges      = []
        self.adj        = {}
        self.mst_edges  = []
        self.sp_path    = []
        self.highlight   = "none"   # "mst" | "sp" | "none"
        self.canvas_widget = None
        self.fig         = None

        self._build_ui()

    # ──────────────────── UI CONSTRUCTION ────────────────────
    def _build_ui(self):
        # ── Title bar ──
        title_bar = tk.Frame(self.root, bg=BG_PANEL, pady=8)
        title_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(title_bar, text="🌾  Farm Irrigation Network Optimization",
                 font=("Consolas", 15, "bold"),
                 fg=ACCENT_GRN, bg=BG_PANEL).pack(side=tk.LEFT, padx=16)

        tk.Label(title_bar,
                 text="Kruskal · Dijkstra · Knapsack",
                 font=("Consolas", 9), fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.RIGHT, padx=16)

        # ── Main layout: left panel + graph area ──
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left control panel
        self._build_control_panel(main_frame)

        # Right: graph canvas + output log
        right_frame = tk.Frame(main_frame, bg=BG_DARK)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_graph_area(right_frame)
        self._build_output_area(right_frame)

    def _build_control_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_PANEL, width=270, padx=14, pady=14)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        def section_label(text):
            tk.Label(panel, text=text, font=FONT_HEAD,
                     fg=ACCENT_GRN, bg=BG_PANEL, anchor="w").pack(fill=tk.X, pady=(10, 2))
            ttk.Separator(panel, orient="horizontal").pack(fill=tk.X, pady=(0, 6))

        # ── Graph generation ──
        section_label("① Graph")
        self._btn(panel, "⟳  Generate Random Graph",
                  ACCENT_GRN, self.cmd_generate)

        # ── Kruskal ──
        section_label("② Kruskal's MST")
        self._btn(panel, "🌿  Run Kruskal's Algorithm",
                  ACCENT_YLW, self.cmd_kruskal, dark_text=True)

        # ── Dijkstra ──
        section_label("③ Dijkstra's Shortest Path")

        src_frame = tk.Frame(panel, bg=BG_PANEL)
        src_frame.pack(fill=tk.X, pady=(0, 6))
        tk.Label(src_frame, text="Source node:", font=FONT_MAIN,
                 fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.LEFT)
        self.src_var = tk.StringVar(value="F1")
        self.src_entry = tk.Entry(src_frame, textvariable=self.src_var,
                                  width=5, bg=BG_CARD, fg=TEXT_MAIN,
                                  insertbackground=TEXT_MAIN,
                                  relief=tk.FLAT, font=FONT_BOLD)
        self.src_entry.pack(side=tk.LEFT, padx=(6, 0))

        tgt_frame = tk.Frame(panel, bg=BG_PANEL)
        tgt_frame.pack(fill=tk.X, pady=(0, 6))
        tk.Label(tgt_frame, text="Target node:", font=FONT_MAIN,
                 fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.LEFT)
        self.tgt_var = tk.StringVar(value="F5")
        self.tgt_entry = tk.Entry(tgt_frame, textvariable=self.tgt_var,
                                  width=5, bg=BG_CARD, fg=TEXT_MAIN,
                                  insertbackground=TEXT_MAIN,
                                  relief=tk.FLAT, font=FONT_BOLD)
        self.tgt_entry.pack(side=tk.LEFT, padx=(6, 0))

        self._btn(panel, "💧  Run Dijkstra's Algorithm",
                  ACCENT_CYN, self.cmd_dijkstra, dark_text=True)

        # ── Knapsack ──
        section_label("④ 0/1 Knapsack")

        cap_frame = tk.Frame(panel, bg=BG_PANEL)
        cap_frame.pack(fill=tk.X, pady=(0, 6))
        tk.Label(cap_frame, text="Water capacity:", font=FONT_MAIN,
                 fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.LEFT)
        self.cap_var = tk.IntVar(value=12)
        tk.Spinbox(cap_frame, from_=5, to=30, textvariable=self.cap_var,
                   width=4, bg=BG_CARD, fg=TEXT_MAIN,
                   buttonbackground=BG_CARD, relief=tk.FLAT,
                   font=FONT_BOLD).pack(side=tk.LEFT, padx=(6, 0))

        self._btn(panel, "🌽  Run Knapsack Algorithm",
                  ACCENT_RED, self.cmd_knapsack)

        # ── Legend ──
        section_label("Legend")
        self._legend_item(panel, ACCENT_GRN, "Graph edges (pipes)")
        self._legend_item(panel, ACCENT_YLW, "MST edges (Kruskal)")
        self._legend_item(panel, ACCENT_CYN, "Shortest path (Dijkstra)")

        # ── Status ──
        self.status_var = tk.StringVar(value="Ready. Generate a graph to begin.")
        tk.Label(panel, textvariable=self.status_var, font=("Consolas", 8),
                 fg=TEXT_DIM, bg=BG_PANEL, wraplength=240,
                 justify=tk.LEFT, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))

    def _btn(self, parent, text, color, command, dark_text=False):
        fg = "#1a2e1a" if dark_text else TEXT_MAIN
        b = tk.Button(parent, text=text, command=command,
                      bg=color, fg=fg, activebackground=color,
                      activeforeground=fg,
                      font=FONT_BOLD, relief=tk.FLAT, cursor="hand2",
                      padx=8, pady=6, anchor="w")
        b.pack(fill=tk.X, pady=3)
        # Hover effect
        b.bind("<Enter>", lambda e, btn=b, c=color: btn.config(bg=self._lighten(c)))
        b.bind("<Leave>", lambda e, btn=b, c=color: btn.config(bg=c))
        return b

    def _lighten(self, hex_color):
        """Return a slightly lighter version of a hex colour."""
        r = min(255, int(hex_color[1:3], 16) + 30)
        g = min(255, int(hex_color[3:5], 16) + 30)
        b = min(255, int(hex_color[5:7], 16) + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _legend_item(self, parent, color, label):
        f = tk.Frame(parent, bg=BG_PANEL)
        f.pack(fill=tk.X, pady=1)
        tk.Canvas(f, width=22, height=4, bg=color, highlightthickness=0).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(f, text=label, font=("Consolas", 8), fg=TEXT_DIM, bg=BG_PANEL).pack(side=tk.LEFT)

    def _build_graph_area(self, parent):
        graph_frame = tk.Frame(parent, bg=BG_DARK, relief=tk.FLAT)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=(8, 8), pady=(8, 4))

        tk.Label(graph_frame, text="Graph Visualization",
                 font=FONT_HEAD, fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w")

        self.graph_container = tk.Frame(graph_frame, bg="#0f1f0f",
                                        relief=tk.FLAT, bd=1)
        self.graph_container.pack(fill=tk.BOTH, expand=True)

        # Placeholder
        tk.Label(self.graph_container,
                 text="Generate a graph to visualize it here.",
                 font=("Consolas", 9), fg=TEXT_DIM, bg="#0f1f0f").pack(expand=True)

    def _build_output_area(self, parent):
        out_frame = tk.Frame(parent, bg=BG_DARK)
        out_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Label(out_frame, text="Algorithm Output",
                 font=FONT_HEAD, fg=TEXT_DIM, bg=BG_DARK).pack(anchor="w")

        self.output = scrolledtext.ScrolledText(
            out_frame, height=9, font=("Consolas", 9),
            bg="#0f1f0f", fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            selectbackground=ACCENT_GRN, selectforeground="#000",
            relief=tk.FLAT, wrap=tk.WORD, padx=8, pady=6)
        self.output.pack(fill=tk.X)

        # Tag colours for styled output
        self.output.tag_config("header",  foreground=ACCENT_GRN,  font=FONT_BOLD)
        self.output.tag_config("kruskal", foreground=ACCENT_YLW,  font=FONT_MAIN)
        self.output.tag_config("dijkstra",foreground=ACCENT_CYN,  font=FONT_MAIN)
        self.output.tag_config("knapsack",foreground=ACCENT_RED,  font=FONT_MAIN)
        self.output.tag_config("info",    foreground=TEXT_DIM,    font=FONT_MAIN)
        self.output.tag_config("bold",    foreground=TEXT_MAIN,   font=FONT_BOLD)

    # ──────────────────── GRAPH DRAWING ────────────────────
    def _draw_graph(self):
        """Render the graph using NetworkX + Matplotlib inside the Tkinter frame."""
        # Clear previous canvas
        for widget in self.graph_container.winfo_children():
            widget.destroy()

        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for w, u, v in self.edges:
            G.add_edge(u, v, weight=w)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=1.5)

        # Categorise edges
        mst_set = {(u, v) for _, u, v in self.mst_edges} | \
                  {(v, u) for _, u, v in self.mst_edges}
        sp_set  = set()
        if len(self.sp_path) > 1:
            for i in range(len(self.sp_path) - 1):
                a, b = self.sp_path[i], self.sp_path[i + 1]
                sp_set.add((a, b)); sp_set.add((b, a))

        normal_edges = [(u, v) for u, v in G.edges()
                        if (u, v) not in mst_set and (u, v) not in sp_set]
        mst_edges_draw = [(u, v) for u, v in G.edges() if (u, v) in mst_set]
        sp_edges_draw  = [(u, v) for u, v in G.edges() if (u, v) in sp_set]

        # Determine node colours
        sp_nodes_set = set(self.sp_path)
        node_colors = []
        for n in G.nodes():
            if n == self.src_var.get() and self.highlight == "sp":
                node_colors.append(ACCENT_CYN)
            elif n == self.tgt_var.get() and self.highlight == "sp":
                node_colors.append("#ff9944")
            elif n in sp_nodes_set and self.highlight == "sp":
                node_colors.append("#2a9a9a")
            else:
                node_colors.append(ACCENT_GRN)

        # Figure
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f1f0f")
        ax.set_facecolor("#0f1f0f")

        # Draw edge layers
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges,
                               edge_color="#3a6a3a", width=1.5,
                               alpha=0.7, ax=ax)
        if mst_edges_draw:
            nx.draw_networkx_edges(G, pos, edgelist=mst_edges_draw,
                                   edge_color=ACCENT_YLW, width=3,
                                   alpha=0.9, ax=ax,
                                   style="solid")
        if sp_edges_draw:
            nx.draw_networkx_edges(G, pos, edgelist=sp_edges_draw,
                                   edge_color=ACCENT_CYN, width=3.5,
                                   alpha=1.0, ax=ax,
                                   style="dashed")

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=650, ax=ax,
                               edgecolors="#0f1f0f", linewidths=2)
        nx.draw_networkx_labels(G, pos, font_color="#0f1f0f",
                                font_size=9, font_weight="bold", ax=ax)

        # Edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_color=TEXT_DIM, font_size=7,
                                     bbox=dict(boxstyle="round,pad=0.2",
                                               fc="#1a2e1a", ec="none", alpha=0.7),
                                     ax=ax)

        # Legend patches
        patches = []
        if mst_edges_draw:
            patches.append(mpatches.Patch(color=ACCENT_YLW, label="MST (Kruskal)"))
        if sp_edges_draw:
            patches.append(mpatches.Patch(color=ACCENT_CYN, label="Shortest Path (Dijkstra)"))
        if patches:
            ax.legend(handles=patches, loc="upper left",
                      facecolor="#1a2e1a", edgecolor="#3a6a3a",
                      labelcolor=TEXT_MAIN, fontsize=7)

        ax.axis("off")
        plt.tight_layout(pad=0.3)

        # Embed in Tkinter
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.graph_container)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig = fig

    # ──────────────────── OUTPUT HELPERS ────────────────────
    def _log(self, text, tag="info"):
        self.output.insert(tk.END, text + "\n", tag)
        self.output.see(tk.END)

    def _log_sep(self):
        self._log("─" * 52, "info")

    def _clear_log(self):
        self.output.delete("1.0", tk.END)

    def _set_status(self, msg):
        self.status_var.set(msg)

    # ──────────────────── COMMAND HANDLERS ────────────────────
    def cmd_generate(self):
        """Generate and display a new random graph."""
        self.nodes, self.edges, self.adj = generate_random_graph()
        self.mst_edges = []
        self.sp_path   = []
        self.highlight  = "none"

        # Update source/target defaults
        self.src_var.set(self.nodes[0])
        self.tgt_var.set(self.nodes[-1])

        self._draw_graph()
        self._clear_log()
        self._log("RANDOM GRAPH GENERATED", "header")
        self._log_sep()
        self._log(f"  Fields (nodes) : {', '.join(self.nodes)}", "info")
        self._log(f"  Pipe count     : {len(self.edges)}", "info")
        self._log_sep()
        self._log("  Edges (pipe connections):", "bold")
        for w, u, v in sorted(self.edges):
            self._log(f"    {u} ──[{w:2d}]── {v}", "info")
        self._set_status(f"Graph: {len(self.nodes)} fields, {len(self.edges)} pipes.")

    def cmd_kruskal(self):
        """Run Kruskal's MST and display results."""
        if not self.nodes:
            messagebox.showwarning("No Graph", "Please generate a graph first.")
            return

        self.mst_edges, total_cost = kruskal(self.nodes, self.edges)
        self.sp_path  = []
        self.highlight = "mst"
        self._draw_graph()

        self._clear_log()
        self._log("KRUSKAL'S MINIMUM SPANNING TREE", "header")
        self._log_sep()
        self._log("  MST edges (minimum-cost irrigation pipes):", "bold")
        for w, u, v in self.mst_edges:
            self._log(f"    {u} ──[{w:2d}]── {v}", "kruskal")
        self._log_sep()
        self._log(f"  Total MST Cost : {total_cost} units", "bold")
        self._log(f"  Edges in MST   : {len(self.mst_edges)}", "info")
        skipped = len(self.edges) - len(self.mst_edges)
        self._log(f"  Edges skipped  : {skipped} (cycle prevention)", "info")
        self._log_sep()
        self._log("  ✔  MST highlighted in YELLOW on the graph.", "kruskal")
        self._set_status(f"Kruskal: MST cost = {total_cost}")

    def cmd_dijkstra(self):
        """Run Dijkstra's shortest path and display results."""
        if not self.nodes:
            messagebox.showwarning("No Graph", "Please generate a graph first.")
            return

        src = self.src_var.get().strip()
        tgt = self.tgt_var.get().strip()

        if src not in self.nodes:
            messagebox.showerror("Invalid Node",
                                 f"Source '{src}' not in graph.\nValid nodes: {', '.join(self.nodes)}")
            return
        if tgt not in self.nodes:
            messagebox.showerror("Invalid Node",
                                 f"Target '{tgt}' not in graph.\nValid nodes: {', '.join(self.nodes)}")
            return

        dist, prev = dijkstra(self.adj, src)
        self.sp_path  = reconstruct_path(prev, src, tgt)
        self.highlight = "sp"
        self._draw_graph()

        self._clear_log()
        self._log("DIJKSTRA'S SHORTEST PATH", "header")
        self._log_sep()
        self._log(f"  Source : {src}   Target : {tgt}", "bold")
        self._log_sep()
        self._log("  Shortest distances from source:", "bold")
        for node in self.nodes:
            d = dist[node]
            bar = "∞" if d == float("inf") else str(d)
            self._log(f"    {node}  →  {bar:>4} units", "dijkstra")
        self._log_sep()
        if self.sp_path:
            path_str = " → ".join(self.sp_path)
            self._log(f"  Shortest path to {tgt}:", "bold")
            self._log(f"    {path_str}", "dijkstra")
            self._log(f"    Distance : {dist[tgt]} units", "dijkstra")
        else:
            self._log(f"  No path found from {src} to {tgt}.", "info")
        self._log_sep()
        self._log("  ✔  Shortest path highlighted in CYAN on the graph.", "dijkstra")
        self._set_status(f"Dijkstra: {src}→{tgt} = {dist.get(tgt, '∞')} units")

    def cmd_knapsack(self):
        """Run 0/1 Knapsack and display crop selection."""
        if not self.nodes:
            messagebox.showwarning("No Graph", "Please generate a graph first.")
            return

        capacity = self.cap_var.get()
        names, weights, values = generate_crops(len(self.nodes))

        max_val, chosen, detail = knapsack_01(capacity, weights, values, names)

        self._clear_log()
        self._log("0/1 KNAPSACK – OPTIMAL WATER DISTRIBUTION", "header")
        self._log_sep()
        self._log(f"  Water Capacity : {capacity} units", "bold")
        self._log_sep()
        self._log("  Available crops (name / water / yield):", "bold")
        for i, (n, w, v) in enumerate(zip(names, weights, values)):
            self._log(f"    [{i+1}] {n:<12} water={w}  yield={v}", "info")
        self._log_sep()
        self._log("  SELECTED crops (optimal combination):", "bold")
        total_w = 0
        for cname, cw, cv in detail:
            self._log(f"    ✔  {cname:<12} water={cw}  yield={cv}", "knapsack")
            total_w += cw
        self._log_sep()
        self._log(f"  Water used     : {total_w} / {capacity}", "bold")
        self._log(f"  Max Yield      : {max_val} units", "bold")
        self._log(f"  Crops selected : {len(detail)}", "info")
        self._set_status(f"Knapsack: max yield = {max_val}, water used = {total_w}/{capacity}")


# ──────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = FarmIrrigationApp(root)
    root.mainloop()
