# MPC Playground

The following notebooks are an experiment in training a model with MPC (https://en.wikipedia.org/wiki/Secure_multi-party_computation)

Solution derived from https://mortendahl.github.io/2017/09/19/private-image-analysis-with-mpc/
and https://iamtrask.github.io/2017/03/17/safe-ai/

See also:
- http://bristolcrypto.blogspot.ca/2016/11/what-is-spdz-part-3-spdz-specifics.html
- https://ec3b8af4-a-62cb3a1a-s-sites.googlegroups.com/site/donbeaver/professional/publications/papers/y1991-c91-Be-cktrand.pdf
- https://github.com/bristolcrypto/SPDZ-2

## Train a Basic Neural Network with MPC

### Description

There are two parties involved in the process: Alice and Bob.

- Alice owns the model weights (which in this case are randomly generated).
- Bob owns the training data.
- Alice never learns the training data.
- Bob never learns the weights.

The weights and the training data are split into shares by their respective owners. The owner keeps a share and sends the other share to the other person. The math required for training the model is applied in a way that keeps the shares secret, however it requires a lot of network communication.

ZeroMQ, http://zeromq.org/, is used to communicate between Alice and Bob. Communication happens synchronously, so both Alice and Bob need to be online and in lockstep.

### Run the Demo

1. Open (Bob's notebook)[./mpc_bob.ipynb] and then `Kernal > Restart and Run All`
2. Open (Alice's notebook)[./mpc_alice.ipynb] and then `Kernal > Restart and Run All`
3. Be patient: By default, the training iterates 10,000 times. There will be about 80,000 messages exchanged between Alice and Bob.
4. After about 50 seconds you should see the resulting weights printed. Verify that Alice and Bob got the same result.
5. Run the (Basic Python Network)[./basic-python-network.ipynb] and the result should be the same (but much faster).

All of the interesting stuff is inside (The SPDZ notebook)[./spdz.ipynb]! From there, you can change the number of training iterations, or see how the cool stuff like matrix multiplication is done with triples and communication between the two parties.

### Next Steps (Request for Contributions!)

- Currently Sigmoid is cheating by combining shares which exposes private data. We should implement `SigmoidInterpolated10` with powering triples to make it secure. @mortendahl 's (post)[https://mortendahl.github.io/2017/09/19/private-image-analysis-with-mpc/] has the details on that.
- Currently the triples are computed on-demand which requires extra network communication, processing, and is not secure. Triples can actually be precomputed by a third party. This would save time in the training process.

### Fun Things to Try

- Currently this uses a basic two layer network. It would be cool to try some more complex models.
- Add, subtract, multiplication, and matrix multiplication are working. After sigmoid and exponents are added, what next?
- Try putting Alice and Bob on different machines across the internet to see what the performance hit is like with real world latency.
