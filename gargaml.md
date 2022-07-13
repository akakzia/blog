---
layout: post
title: "GARGAML: Addressing Multitask and Multigoal Transfer with Goal-Conditioned Meta Reinforcement Learning and Replay"
description: "A graphic designer is a professional within the graphic design and graphic arts industry."
date: 2022-07-13
---

> When tasks and goals are not known in advance, an agent may use either multitask learning or meta reinforcement learning to learn how to transfer knowledge from what it learned before. Recently, goal-conditioned policies and hindsight experience replay have become standard tools to address transfer between goals in the multitask learning setting. In this paper, we show that these tools can also be imported into the meta reinforcement learning when one wants to address transfer between tasks and between goals at the same time. More importantly, we investigate whether the meta reinforcement learning approach brings any benefit with respect to multitask learning in this specific context. Our study based on a basic meta reinforcement learning framework reveals that showcasing such benefits is not straightforward, calling for further comparisons between more advanced frameworks and richer sets of tasks. 

<!--- Introduction -->

In an open-ended learning context {%cite doncieux2018open %}, an agent faces a continuous stream of unforeseen tasks along its lifetime and must learn how to accomplish them.
For doing so, two closely related reinforcement learning (*RL*) frameworks are available: multitask learning (*MTL*) and meta reinforcement learning (*MRL*).

Multitask learning was first defined in {%cite caruana97multitask %}. The general idea was to train a unique parametric policy to solve a finite set of tasks so that it would finally perform well on all these tasks. Various *MTL* frameworks have been defined since then {%cite yang2014unified %}. For instance, an *MTL* agent may learn several policies sharing only a subset of their parameters so that **transfer learning** can occur between these policies {%cite taylor2009transfer %}.
Multitask learning is the matter of an increasing research effort since the advent of powerful *RL* methods {%cite florensa2018automatic veeriah2018many ghosh2018learning %}.

<p align = "center"><img src = "/images/task_goal.png" width="600"></p><p align = "center">
Fig.1 - The Fetch Environment with different tasks and goals.
</p>

But this effort came with a drift from the multitask to the multigoal context. Actually, the distinction between tasks and goals is not always clear. In this paper we will rely on an intuitive notion illustrated in *Figure 1* of task as some abstract activity such as **em push blocks** or **stack blocks**, and we define a goal as some concrete state of the world that an agent may want to achieve given its current task, such as **pick block "A" and place it inside this specific area**. If we stick to these definition, a lot of work pretending to transfer between tasks actually transfer between goals. Anyways, for multigoal learning, goal-conditioned policies (*GC-P*) have emerged as a satisfactory framework to represent a set of policies to address various goals, as it naturally provides some generalization property between these goals. Besides, the use of Hindsight Experience Replay (*HER*) {%cite andrychowicz2017hindsight %} has been shown to significantly speed up the process of learning to reach several goals when the reward is sparse.
In this context, works trying to learn several tasks and several goals at the same time are just emerging (see e.g. Table 1 in {%cite colas2019curious %}).

Meta reinforcement learning is a broader framework. Generally speaking, it consists in using inductive bias obtained from learning a set of policies, so that new policies for addressing similar unknown tasks can be learned in only a few gradient steps. In this latter process called {\em fine tuning}, policy parameters are tuned specifically to each new task \citep{rakelly2019efficient}.
The \mrl framework encompasses several different perspectives \citep{weng2019metaRL}.
One consists in learning a recurrent neural network whose dynamics update the weights of a policy so as to mimic a reinforcement learning process \citep{duan2016rl,wang2016learning}. This approach is classified as {\em context-based} meta-learning in \cite{rakelly2019efficient} as the internal variables of the recurrent network can be seen as latent context variables.
Another perspective is efficient parameter initialization
\citep{finn2017model}, classified as {\em gradient-based} meta-learning. Here, we focus on a family of algorithms encompassing the second perspective which started with Model-Agnostic Meta-Learning (\maml) \citep{finn2017model}. A crucial feature of this family of algorithms is that they introduce a specific mechanism to favor transfer between tasks, by looking for initial policy parameters that are close enough to the manifold of optimal policy parameters over all tasks.

Actually, though in principle \mrl is more meant to address several tasks than several goals in the same tasks, in practice empirical studies often address the same benchmarks which are multigoal rather than multitask \cite{finn2017model,rothfuss2018promp,rakelly2019efficient}. Only a few papers truly transfer knowledge from one task to another \citep{zhao2017tensor,colas2019curious,fournier2019clic_arxiv}, but at least the latter two do not classify themselves as performing \mrl.

The common practice in \mtl consists in sampling from all target tasks, which limits its applicability to the open-ended learning context, where some target tasks may not be known in advance.
Besides, transfer between tasks in \mtl, or rather between goals, usually relies on the generalization capability of the underlying function approximator, without any specific mechanism to improve it, resulting in potential brittleness when this approximator is not appropriate.
Nevertheless, when applied to the multigoal setting using \gcps and \her, this approach has been shown to provide efficient transfer capabilities.

By contrast, in \mrl, the test tasks are left apart during training.
To ensure good transfer to these unseen tasks, the approach in \maml consists in iteratively refining the initial policy parameters so that a new task can be learned through only a few gradient steps. In principle, this latter transfer-oriented mechanism makes \mrl more appropriate for open-ended learning, where the agent cannot train in advance on future unknown tasks.

The goal of this paper is to show that the above multigoal and multitask learning mechanisms can be combined into a single learning framework addressing open-ended learning when each task can be instantiated with multiple goals. By doing so, we also clarify the relationship between these mechanisms and show that they are more complementary than competing.
%However, the use of \gcps may also endow some \mtl systems with the capability to generalize over unseen goals without having to rely on additional policy tuning steps.
%Furthermore, early \mrl frameworks like \maml are reported to be slow and sample inefficient \citep{antoniou2018train,rakelly2019efficient}, which may speak in favor of using \mtl rather than \mrl.

%In this paper, we investigate all the above questions, providing a more up-to-date comparison between \mtl and \mrl.

The main contributions of the paper are the following:
+ We design a new \maml-like \mrl algorithm named \gargaml\footnote{for "Goal-Aware Replay in Generalization-Apt Meta-Learning"}. *GARGAML* is based on the Soft Actor Critic (*SAC*) algorithm \citep{haarnoja2018soft}, a sample efficient off-policy *RL* algorithm, it incorporates *GCPS* and *HER* to deal with multiple goals, and uses the specific transfer-oriented mechanism of *MAML* to deal with multiple tasks. We show how each of these components contributes to the better performance of the global system in a series of benchmarks.

+ We show that, when applied to a single task with multiple goals, the specific transfer-oriented mechanism of \maml does not bring any benefit with respect to just using \gcps and \her, even when clearly separating training goals from test goals.

+ By contrast, we show that the same mechanism plays a crucial role for transfer between training tasks and test tasks, opening the way to efficient multigoal open-ended learning.


<!--- End Introduction -->

<!--- References -->

{% bibliography --cited %}
