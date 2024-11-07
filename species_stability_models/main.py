from source import * 

if __name__ == '__main__':

    print("Select the simulation type:")
    print("1. Random ")
    print("2. Predator-Prey ")
    print("3. Competition-Mutualistic ")
    print("4. Mutualistic ")

    selection = input("\nEnter the number corresponding to the simulation type: ")
    simulation_functions = {
        '1': create_random_matrix,
        '2': create_pred_prey_matrix,
        '3': create_competition_mutualism_matrix,
        '4': create_mutualism_matrix
    }
    selected_function = simulation_functions.get(selection)

    if selected_function is None:
        print("Invalid selection.")
    else:
        create_simulation(selected_function)