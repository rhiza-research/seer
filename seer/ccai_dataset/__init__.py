"""The CCAI dataset: many perturbed versions of each final seer instance, some of them solved.

    python -m seer.ccai_dataset.run seer/ccai_dataset/configs/<country>.yaml

run.py generates and solves (see its docstring for the output layout); provenance.py records how each
perturbation was made and regenerates one from its record. One config per country in configs/.
"""
