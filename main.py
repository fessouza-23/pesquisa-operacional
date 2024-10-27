from functions import create_model, solve_model, get_results, print_schedule, plot_schedule

# Unidade Centro
model_centro, C_centro, training_data_centro = create_model("centro")
status_centro = solve_model(model_centro)
if status_centro == 1:
    print("Solução para Unidade Centro encontrada com sucesso!")
    schedule_centro = get_results(C_centro, training_data_centro)
    print_schedule(schedule_centro)
    plot_schedule(schedule_centro, "centro_schedule.png", "centro")
    print("Cronograma da Unidade Centro salvo como 'centro_schedule.png'")
else:
    print("Não foi possível encontrar uma solução para a Unidade Centro.")

# Unidade Compacta
model_compacta, C_compacta, training_data_compacta = create_model("compacta")
status_compacta = solve_model(model_compacta)
if status_compacta == 1:
    print("Solução para Unidade Compacta encontrada com sucesso!")
    schedule_compacta = get_results(C_compacta, training_data_compacta)
    print_schedule(schedule_compacta)
    plot_schedule(schedule_compacta, "compacta_schedule.png", "compacta")
    print("Cronograma da Unidade Compacta salvo como 'compacta_schedule.png'")
else:
    print("Não foi possível encontrar uma solução para a Unidade Compacta.")
