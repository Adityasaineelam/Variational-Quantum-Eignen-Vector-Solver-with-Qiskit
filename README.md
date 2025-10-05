# Variational-Quantum-Eignen-Vector-Solver-with-Qiskit
Variational Quantum Eigensolver (VQE) implemented in Qiskit to estimate ground state energies of quantum systems using hybrid quantum-classical optimization.
import matplotlib.pyplot as plt
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.settings import settings

# Use sparse matrix solver
settings.use_pauli_sum_op = False

# Define bond lengths to simulate
bond_lengths = [0.5, 0.735, 1.0, 1.5, 2.0]
energies = []

# Loop over bond lengths
for dist in bond_lengths:
    atom_string = f"H .0 .0 .0; H .0 .0 {dist}"
    driver = PySCFDriver(atom=atom_string, basis="sto3g")
    es_problem = driver.run()

    transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
    problem = transformer.transform(es_problem)

    mapper = JordanWignerMapper()
    main_op = problem.second_q_ops()[0]
    qubit_op = mapper.map(main_op)

    ansatz = UCCSD(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
    )

    estimator = Estimator()
    optimizer = COBYLA(maxiter=100)
    vqe_solver = VQE(estimator, ansatz, optimizer)

    solver = GroundStateEigensolver(mapper, vqe_solver)
    result = solver.solve(problem)

    energy = result.total_energies[0].real
    energies.append(energy)
    print(f"Bond length {dist} Å → Energy: {energy:.6f} Hartree")

#  Plotting the energy vs bond length
plt.figure(figsize=(8, 5))
plt.plot(bond_lengths, energies, marker='o', linestyle='-', color='teal')
plt.xlabel("Bond Length (Å)")
plt.ylabel("Ground State Energy (Hartree)")
plt.title("H₂ Ground State Energy vs Bond Length")
plt.grid(True)
plt.tight_layout()
plt.show()
