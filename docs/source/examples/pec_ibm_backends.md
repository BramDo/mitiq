---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{tags} pec, qiskit, basic
```
# Probabilistic error cancellation on IBM Quantum backends with Mirror Circuits

This tutorial demonstrates how to perform probabilistic error cancellation (PEC) on IBM backends.
We assume a depolarizing noise model and show how to obtain the relevant error
information from a Qiskit fake backend. The same approach can be applied to a
real backend by using ``QiskitRuntimeService``.

## Settings

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.fake_provider import FakeSherbrookeV2

from mitiq import pec, benchmarks
from mitiq.interface.mitiq_qiskit.conversions import from_qiskit
```

Set ``USE_REAL_HARDWARE`` to ``True`` to run on a real device.

```{code-cell} ipython3
USE_REAL_HARDWARE = False
```

**Note:** When ``USE_REAL_HARDWARE`` is set to ``False`` a fake backend is used
instead of a real quantum processor.

## Selecting the backend

```{code-cell} ipython3
if USE_REAL_HARDWARE and QiskitRuntimeService.saved_accounts():
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False)
else:
    backend = FakeSherbrookeV2()
```

### Fetching device information

```{code-cell} ipython3
coupling_map = backend.coupling_map
properties = backend.properties()

cnot_error = None
for gate in properties.gates:
    if gate.gate == "cx" and gate.qubits == [0, 1]:
        cnot_error = [p.value for p in gate.parameters if p.name == "gate_error"][0]
        break

coupling_map, cnot_error
```

The coupling map details which qubits are connected. ``cnot_error`` provides the
error probability for the ``CX(0, 1)`` operation.

### Building PEC representations

```{code-cell} ipython3
qc = QuantumCircuit(2)
qc.cx(0, 1)

mitiq_circuit = from_qiskit(qc)
representation = pec.represent_operation_with_global_depolarizing_noise(
    mitiq_circuit,
    noise_level=cnot_error,
)
```

The ``representation`` object stores the quasiprobability description of the
ideal CNOT gate under depolarizing noise.

## Mirror circuit benchmark

```{code-cell} ipython3
graph = nx.Graph(coupling_map)
depth = 3

circuit, correct_bitstring = benchmarks.generate_mirror_circuit(
    nlayers=depth,
    two_qubit_gate_prob=1.0,
    connectivity_graph=graph,
    two_qubit_gate_name="CNOT",
    seed=0,
    return_type="qiskit",
)
```

### Running the experiment

```{code-cell} ipython3
def execute(circ, back, shots, bitstring):
    job = back.run(circ, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts.get(bitstring, 0) / shots
```

```{code-cell} ipython3
shots = 1024
executor = lambda c: execute(c, backend, shots, "".join(map(str, correct_bitstring)))

pec_value = pec.execute_with_pec(
    circuit,
    executor,
    representations=[representation],
    num_samples=50,
)
pec_value
```

When ``USE_REAL_HARDWARE`` is ``True`` the same workflow applies after
initializing ``backend`` with ``QiskitRuntimeService``.
