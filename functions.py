from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpContinuous, LpBinary
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_model(unit):
    if unit == "centro":
        num_clients = 6
        num_machines = 6
        M = 100000 
        training_data = {
            1: [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
            2: [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
            3: [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
            4: [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
            5: [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
            6: [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)]
        }
    elif unit == "compacta":
        num_clients = 10
        num_machines = 5
        M = sum(max(time for (machine, time) in client_data) for client_data in [
            [(1, 21), (0, 53), (4, 95), (3, 55), (2, 34)],
            [(0, 21), (3, 52), (4, 16), (2, 26), (1, 71)],
            [(3, 39), (4, 98), (1, 42), (2, 31), (0, 12)],
            [(1, 77), (0, 55), (4, 79), (2, 66), (3, 77)],
            [(0, 83), (3, 34), (2, 64), (1, 19), (4, 37)],
            [(1, 54), (2, 43), (4, 79), (0, 92), (3, 62)],
            [(3, 69), (4, 77), (1, 87), (2, 87), (0, 93)],
            [(2, 38), (0, 60), (1, 41), (3, 24), (4, 83)],
            [(3, 17), (1, 49), (4, 25), (0, 44), (2, 98)],
            [(4, 77), (3, 79), (2, 43), (1, 75), (0, 96)]
        ])
        training_data = {
            1: [(1, 21), (0, 53), (4, 95), (3, 55), (2, 34)],
            2: [(0, 21), (3, 52), (4, 16), (2, 26), (1, 71)],
            3: [(3, 39), (4, 98), (1, 42), (2, 31), (0, 12)],
            4: [(1, 77), (0, 55), (4, 79), (2, 66), (3, 77)],
            5: [(0, 83), (3, 34), (2, 64), (1, 19), (4, 37)],
            6: [(1, 54), (2, 43), (4, 79), (0, 92), (3, 62)],
            7: [(3, 69), (4, 77), (1, 87), (2, 87), (0, 93)],
            8: [(2, 38), (0, 60), (1, 41), (3, 24), (4, 83)],
            9: [(3, 17), (1, 49), (4, 25), (0, 44), (2, 98)],
            10: [(4, 77), (3, 79), (2, 43), (1, 75), (0, 96)]
        }

    model = LpProblem(f"{unit.capitalize()}_Gym_Scheduling", LpMinimize)

    C = LpVariable.dicts("C", [(i, k) for i in range(1, num_clients + 1) for k in range(num_machines)], lowBound=0, cat=LpContinuous)
    x = LpVariable.dicts("x", [(i, j, m) for i in range(1, num_clients + 1) for j in range(1, num_clients + 1) if i != j for m in range(num_machines)], cat=LpBinary)

    model += lpSum([C[(i, num_machines - 1)] for i in range(1, num_clients + 1)])

    for i in range(1, num_clients + 1):
        for k in range(len(training_data[i]) - 1):
            current_machine, current_time = training_data[i][k]
            next_machine, next_time = training_data[i][k + 1]
            model += C[(i, next_machine)] >= C[(i, current_machine)] + current_time, f"Sequencing_{unit}_{i}_{k}"

    for m in range(num_machines):
        for i in range(1, num_clients + 1):
            for j in range(1, num_clients + 1):
                if i != j:
                    machine_time_i = dict(training_data[i]).get(m, None)
                    machine_time_j = dict(training_data[j]).get(m, None)
                    if machine_time_i is not None and machine_time_j is not None:
                        model += C[(j, m)] >= C[(i, m)] + machine_time_i - M * (1 - x[(i, j, m)])
                        model += C[(i, m)] >= C[(j, m)] + machine_time_j - M * x[(i, j, m)]

    for m in range(num_machines):
        for i in range(1, num_clients + 1):
            for j in range(i + 1, num_clients + 1):
                model += x[(i, j, m)] + x[(j, i, m)] == 1

    return model, C, training_data

def solve_model(model):
    model.solve()
    return model.status

def get_results(C, training_data):
    schedule = {i: [] for i in training_data.keys()}
    for i in training_data.keys():
        for machine, time in training_data[i]:
            start_time = C[(i, machine)].varValue
            schedule[i].append((machine, start_time, time))
    return schedule

def print_schedule(schedule):
    for client, tasks in schedule.items():
        print(f"Cliente {client}:")
        for machine, start_time, duration in sorted(tasks, key=lambda x: x[1]):
            print(f"  Aparelho {machine}: Início em {start_time:.2f} min, Duração de {duration} min")
        print()

def plot_schedule(schedule, filename, unit):
    machine_colors = {
        0: "#66c2a5",
        1: "#fc8d62",
        2: "#8da0cb",
        3: "#e78ac3",
        4: "#ffd92f",
    }
    if unit == "centro":
        machine_colors[5] = "#b3b3cc"

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlabel("Tempo (minutos)")
    ax.set_ylabel("Clientes")
    ax.set_title(f"Cronograma de Treinamento - {filename.replace('.png', '').capitalize()}")

    y_ticks = [i * 10 + 5 for i in range(len(schedule))]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"C{i}" for i in schedule.keys()])

    for i, (client, tasks) in enumerate(schedule.items()):
        y = i * 10
        for machine, start_time, duration in tasks:
            color = machine_colors.get(machine, "#ffd92f")
            ax.broken_barh([(start_time, duration)], (y, 8), facecolors=color)
            ax.text(start_time + duration / 2, y + 4, f"A{machine}\n{int(start_time)}-{int(start_time + duration)}",
                    ha='center', va='center', color='black', fontsize=8, fontweight='bold')

    legend_patches = [mpatches.Patch(color=color, label=f"Aparelho {machine}") for machine, color in machine_colors.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Aparelhos")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)