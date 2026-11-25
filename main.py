
import csv
import heapq
from collections import deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "roads.csv"


def load_graph():
    graph = {}

    with open(DATA_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            source = row["from"]
            destination = row["to"]
            cost = float(row["travel_cost"])

            graph.setdefault(source, [])
            graph.setdefault(destination, [])

            graph[source].append((destination, cost))
            graph[destination].append((source, cost))

    return graph


def bfs(graph, start):
    visited = set()
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()

        if node not in visited:
            visited.add(node)
            result.append(node)

            for neighbour, _ in graph[node]:
                if neighbour not in visited:
                    queue.append(neighbour)

    return result


def dfs(graph, start):
    visited = set()
    result = []

    def visit(node):
        visited.add(node)
        result.append(node)

        for neighbour, _ in graph[node]:
            if neighbour not in visited:
                visit(neighbour)

    visit(start)
    return result


def dijkstra(graph, start, end):
    distances = {node: float("inf") for node in graph}
    previous = {}

    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        cost, node = heapq.heappop(priority_queue)

        if cost > distances[node]:
            continue

        if node == end:
            break

        for neighbour, weight in graph[node]:
            candidate = cost + weight

            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = node

                heapq.heappush(
                    priority_queue,
                    (candidate, neighbour)
                )

    if distances[end] == float("inf"):
        return float("inf"), []

    path = []
    current = end

    while current != start:
        path.append(current)
        current = previous[current]

    path.append(start)
    path.reverse()

    return distances[end], path


def prim_mst(graph, start):
    visited = {start}
    edges = []
    total_cost = 0
    selected_edges = []

    for neighbour, weight in graph[start]:
        heapq.heappush(
            edges, (weight, start, neighbour)
        )

    while edges and len(visited) < len(graph):
        weight, source, destination = heapq.heappop(edges)

        if destination in visited:
            continue

        visited.add(destination)
        selected_edges.append(
            (source, destination, weight)
        )
        total_cost += weight

        for neighbour, edge_weight in graph[destination]:
            if neighbour not in visited:
                heapq.heappush(
                    edges,
                    (edge_weight, destination, neighbour)
                )

    if len(visited) != len(graph):
        raise ValueError("Graph is disconnected")

    return selected_edges, total_cost


def main():
    graph = load_graph()

    start = "A"
    destination = "H"

    print("=" * 50)
    print("SMART CITY TRAFFIC MANAGEMENT SYSTEM")
    print("=" * 50)

    print("\nLocations:", ", ".join(sorted(graph)))

    print("\nBFS Traversal:")
    print(" -> ".join(bfs(graph, start)))

    print("\nDFS Traversal:")
    print(" -> ".join(dfs(graph, start)))

    cost, path = dijkstra(graph, start, destination)

    print("\nShortest Route:")
    print(" -> ".join(path))
    print("Total Cost:", cost)

    mst_edges, mst_cost = prim_mst(graph, start)

    print("\nMinimum Spanning Tree:")

    for source, destination, weight in mst_edges:
        print(f"{source} -- {destination}: {weight}")

    print("Total MST Cost:", mst_cost)


if __name__ == "__main__":
    main()
