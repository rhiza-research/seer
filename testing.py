from synthseer.build_model.build_model import build_model
from gams2pypsa.process_generator_data import process_generator_data

if __name__ == "__main__":
    #model = build_model()
    #model.optimize()
    #print(model.objective)
    process_generator_data("gams2pypsa/data/processed/Gambia/")
    import pdb; pdb.set_trace()