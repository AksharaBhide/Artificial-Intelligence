# Graph connections
graph = {
    'A': {'B': 1, 'C': 3},
    'B': {'D': 3},
    'C': {'D': 1, 'E': 2},
    'D': {'G': 5},
    'E': {'G': 2},
    'G': {}
}

# Heuristic values (distance to goal G)
heuristics = {
    'A': 6,
    'B': 5,
    'C': 4,
    'D': 3,
    'E': 1,
    'G': 0
}

def solve_path_astar():
    
    start = input("Enter start node (A-G): ").upper()
    goal = input("Enter goal node (A-G): ").upper()

    open_list = [[start, heuristics[start]]]
    closed_list = []

    g_costs = {start: 0}
    parents = {start: None}

    while len(open_list) > 0:

        open_list.sort(key=lambda x: x[1])
        n_data = open_list.pop(0)
        n = n_data[0]

        if n == goal:
            print("\nSUCCESS!! Path found.")
            show_result(parents, goal, g_costs[goal])
            return

        closed_list.append(n)

        for neighbor, cost in graph[n].items():

            new_g = g_costs[n] + cost
            new_f = new_g + heuristics[neighbor]

            if neighbor not in closed_list and neighbor not in [item[0] for item in open_list]:

                g_costs[neighbor] = new_g
                open_list.append([neighbor, new_f])
                parents[neighbor] = n

            else:
                if new_g < g_costs.get(neighbor, 9999):

                    g_costs[neighbor] = new_g
                    parents[neighbor] = n

                    for item in open_list:
                        if item[0] == neighbor:
                            item[1] = new_f

    print("FAILURE: No path found.")

def show_result(parents, goal, total_cost):

    path = []
    curr = goal

    while curr is not None:
        path.append(curr)
        curr = parents[curr]

    path.reverse()

    print("Path:", " -> ".join(path))
    print("Total Cost:", total_cost)


# Run program
solve_path_astar()