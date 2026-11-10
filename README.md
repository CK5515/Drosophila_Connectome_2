# Fruit flies *still* might think in Spectral MoE...
![3d render of a fly](./Images/flypoly.jpg)
### Prerequisite [here!!!](https://github.com/CK5515/Drosophila_Connectome)   

Well I hope you read the prerequisite cuz I am so back. Last attempt was a lil humiliating but I published it anyways. I valued it from a scientific and learning perspective. It was a stupid exploration and I made many mistakes and bad assumptions. Perhaps I will focus less on hypothesizing and more on executing based on evidence. I'm not getting beaten by this fly and WILL arive at a conclusion. Taking lessons from last time is quite nice and I can work away imidiately. Ofc my SWAEV compute is still onset so I have free time. This is more than just an exploration. I need this question answered. Do fly brains actually think in MORGAN style spectral MoE??? 
<hr/>

## 0.

*Rules...*
<hr/>

Originally my biggest mistake was writing the rules while I was already looking at results. So before a single part 2 run I wrote [the plan](./docs/plan.md) and locked in. It has the rules, seeds (5 now) and what every outcome means.

New rules on top of part 1's:

+ **floor gate**: the model has to beat "predict each neuron's average" before ANY MoE claim. Part 1 sat at -1.59 against 0.145 and I still tried to interpret things. That was silly of me.
+ **specialisation**: frozen experts can't count anymore. An expert has to have moved from init, matter when knocked out and differ in timescale by 2 times or more.
+ **routing**: the router has to actually use the input, measured as mutual information between which expert wins and what the stimulus was.
+ **Positive control**: a "no" only counts if the pipeline says "yes" when the answer is yes...so it's data driven basically

I got compute too! MeluXina, 4 times A100 per node, ~700 node hrs over 3 months (shared with another programme).

## 1.

*Did the fly even have anything to route?*
<hr/>

Before spending a single cluster hour I went back to part 1's data. Does the LIF teacher actually *have* slow, ringy dynamics for damped-wave experts to fit?

Kinda no. All of the teacher's dynamics happen within one or two 20 ms bins. Fast rise, a small overshoot (MN9 overshoots about 12-15%)... then flat. After the stimulus turns off, activity drops to 42% in one bin and... silence. Nothing is slower than ~40 ms and nothing rings. The residual spectrum just rises from 2 to 25 Hz with no peak.

So part 1's "learned" expert timescales of 30-81 ms? The init γ = 0.3 works out to ≈ 67 ms. Most experts never moved from init. Those numbers were mostly the init, not anything they learned...

`runs/a0_teacher_resonance/20261010-180821`

## 2.

*Is the supercomputer the same fly?*
<hr/>

The cluster runs torch 2.5.1 on A100s, while part 1 ran torch 2.2.2 on my 3060. Before comparing anything to part 1, I reran part 1's runs as an anchor.

moe_full -1.70 (part 1: -1.59 ± 1.25), mlp 0.846 (0.845), mean 0.14491 (identical to five figures). Two seeds landed within 0.006 of part 1 and one moved by 0.31. Close enough to share a table but not bit-identical.

## 3.

*Was it just undertrained?*
<hr/>

Part 1 hit the 40 epoch cap everywhere, so the first real job was training to convergence with a plateau schedule. All 12 runs stopped early with the learning rate cut to basically zero.

moe_full -1.69 ± 0.90. moe_linear actually got *worse* (-4.09). chebgru improved a lot (-0.31 / -0.17). Floor gate still fails. What bout the router? In 9 of 10 layers it's one-hot and identical for every stimulus. Unused experts still sit at their 67 ms init. Not epic.

So no... the epoch cap was not the reason. 10.9 node hrs.

## 4.

*Would I even see it if it was there?*
<hr/>

This is the positive control. I built a fake fly... a teacher made from my own SpectralMoE on the real connectome basis with four known experts at 20/50/125/315 ms and a fixed router that depends on which taste channels are on. If my pipeline can't find structure I literally planted, nothing it says about the real fly means anything. Plus a planted null where the rules must stay silent.

First I scored the *true* teachers through my rules before training anything and specialisation failed on the planted truth itself. Fixed the rule (pooled knockouts) and strengthened the planted teacher. All before any student trained, all logged.

Then the students...

4 of 5 seeds stalled at the floor with one-hot routers. Part 1's collapse reproduced. One seed fit beautifully (0.939). Therefore it is an optimisation problem.

It also caught three problems with my approach:

+ floor gate used the mean over seeds, so one good seed could carry four bad ones.
+ specialisation could pass on a seed that didn't even fit (ratio 3.5 at a score of 0.074).
+ The timescales I was reading out aren't identifiable from the data. Two layers compose and α trades off against γ. Even the seed that fit read 22/28/52/170 against the planted 20/50/125/315... right order squashed together.

So part 1's pipeline could never have succeeded...

## 5.

*Fixing the router*
<hr/>

Why did the routers collapse? I poked at the checkpoints. At init the gate is nicely spread. Every collapsed seed ends with a gap of ~30 between the top two logits... so the softmax gradient is about e^-30. It cannot come back. Meanwhile the gate's input features drifted from 3-6 up to 7-13.

I tried to build a better timescale recovery probe. It failed on the true teachers too so recovery became descriptive only. I wrote that decision down before any result came in.

**PC passes.** Planted 0.85-0.87 in all five seeds (was 1/5). specialisation 5/5, routing 5/5. Routing usage close to the true router's. The null stays silent on specialisation...

**PLEASE READ!!!!**

*...but routing on the null was thin. 3 of 5 null seeds passed routing with near-uniform routers because the argmax of a near-uniform gate still tracks the stimulus a little. So I tightened routing (MI ≥ 0.3 nats, entropy < 0.9). Those thresholds were set from the gap between planted and null, NOT a prior. The planted ones all read ≥ 0.62 and the null ≤ 0.18, so the line sits in a wide gap. Null silent 0/5 under v3.*

## 6.

*Round 1*
<hr/>

Using the same calibrated setup and real fly data.

**Floor gate fails 5/5.** -3.0 to -3.8, worse than without the router fixes. The router no longer collapses... but two experts carry everything and the second one is overdamped.

## 7.

*Why is it so bad tho?*
<hr/>

It's not overfitting... train and test agree. Part 1's silent-neuron trap strikes again! Now I know its tricks...

+ The worst 1% of neurons make up ~70% of the negative score.
+ Variance-weighted, moe_full fits **0.75** of the signal and the top variance decile scores 0.99.
+ The cause being **the loss weights neurons by variance, while floor gate weights every neuron equally.** A model that reads out through shared eigenmodes pays for that and leaks activity into quiet neurons. The MLP has its own output per neuron, so it doesn't care.

So a new arm (A6) with per-neuron loss weighting, built from the train split only. The PC went first again... it **passes, even better** (planted 0.92-0.93, all four experts carrying weight, null silent).

Then the fly: **0.07-0.11 against a floor of 0.145.** From -3.4 to just under the line. Close... but floor gate fails 5/5 again. And the router is near-uniform, which is the *null's* pattern, not the planted one.

**So the static taste task is done.** The tool can detect routing when it's there. On this task it sees a model that almost fits, with a router that doesn't route. OK. I mean it's only 5 nrs not much for it to do...

## 8.

*New task*
<hr/>

NEW PHASE B gives the router a real job. Three circuits instead of one and inputs that change over time.

Switched to the FlyWire **v783** connectome since the public cell-type annotations only use v783 IDs. Checked the teacher still behaves... sugar to MN9 dose response is close to v630.

Introducing **the kill gate.** I pulse each input population and record the whole brain at 1 ms. If every circuit has the same timescale (within 1.5 times) and nothing rings, timescale specialisation is impossible in this teacher and the specialisation answer is decided before training.

It doesn't trip. Feeding (sugar to MN9) is the slow one at ~12 ms. Grooming (Johnston's organ to descending neurons) and escape (LC4/LPLC2 to giant fiber) are 6-11 ms. Spread is 1.96 times at the long drive...

*...but by a 2 times margin... not so wide. The "ringing" turned out to be output neurons firing spike trains phase-locked to the pulse, not the network oscillating. My ringing test can't tell those apart. Everything responds within 5-35 ms and is silent by ~70 ms. There's something to separate. Just barely... all of it is fast.* Which means 2 ms bins is the method.

This is the new circuit. Feeding + grooming + escape, same k-hop carve as part 1 with a cap per circuit... **8,677 neurons, 997k edges, 46% inhibitory.** `runs/b1_circuits/20261110-100042`
