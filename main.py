from synthseer.data import synth_data
from synthseer.model import create_model_from_data
from synthseer.visualize import viz_model

def main2():
    input_data = synth_data(2025, nyears=3, load_growth=0.2, rez="24h", excess_cap=1)

def main():
    # create input data - 3 years, 24h resolution, load growth 0.2, excess cap 1
    input_data = synth_data(2025, nyears=7, load_growth=0.2, rez="24h", excess_cap=1)
    # create model
    model = create_model_from_data(input_data)
    # Optimize model
    model.optimize()
    # Visualize model
    import pdb; pdb.set_trace()
    viz_model(model)


if __name__ == "__main__":
    main()
