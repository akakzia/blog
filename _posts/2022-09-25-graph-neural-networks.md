---
layout: post
title: "When Graph Neural Networks Meet Reinforcement Learning"
description: "An overview on Graph Neural Networks in Reinforcement Learning."
feature_image: images/graphs.png
date: 2022-09-25
---

>Reinforcement Learning (RL) agents should be able to efficiently generalize to novel situations and transfer their learned skills. Without these properties, such agents would always have to learn from scratch, even though they have already mastered primitive skills that could potentially be leveraged to acquire more complex ones. 

<!--more-->

Combining primitive skills and building upon them to solve harder tasks is a key challenge within artificial intelligence. In the context of _goal-conditioned agents_, transfer and adaptibility seem to depend on two key features: _the goal space design_, and _the policy architecture_. On the one hand, the goal representation---whether it is learned or predefined---should encapsulate an adequate structure that defines a specific topology in the goal space. 

On the other hand, since the behavior of artificial agents does not only depend on how they represent their goals, but also on how they take actions, we investigate _Graph Neural Networks_ (GNNs) as technical tools to model policies in autotelic agents. This choice is also motivated by developmental approaches, as research in psychology shows that humans perceive their world in a structured fashion {%cite winston1970learning palmer1975visual navon1977forest markman1989categorization kemp2008discovery tenenbaum2011grow battaglia2016interaction battaglia2018relational godfrey2021theory %}.

This blog post is organized as follows. First, we start by introducing GNNs as technical tools to endow artificial agents with relational inductive biases. Then, we present an overview on the use of GNNs in the field of RL. Finally, we highlight several limitations of such combination. 

## Graph Neural Networks

Recently, deep learning methods have been used to solve a significant amount of problems in different domains. Ranging from image classification {%cite redmon2016you ren2015faster %} and video processing {%cite zhang2016deep %} to speech recognition {%cite hinton2012deep %} and neural machine translation {%cite luong2015effective wu2017sequence %}, these methods use parameterized neural networks as building blocks. Consequently, such methods are usually end-to-end, requiring few to no assumptions. They feed their networks with raw streams of data which are usually represented in the Euclidean Space. However, many applications rather represent data in non-Euclidean domains and use graphs with complex relationships and inter-dependencies. Standard usage of deep learning techniques usually struggle with this type of unstructured representations.

Interestingly, research has been interested in leveraging graph-based information using neural networks. Namely, _Graph Neural Networks_ (GNNs) were proposed as computational frameworks that handle unstructured data using neural networks that they share between nodes and edges {%cite wang2016learning battaglia2016interaction santoro2017simple zaheer2017deep hamrick2017metacontrol sanchez2018graph battaglia2018relational zambaldi2018relational wang2018nervenet bapst2019structured li2019towards colas2020language akakzia2021grounding akakzia2022learning %}. Although these methods are all based on the same idea, they use different techniques depending on how they handle computations within their GNNs' definition. There exist several surveys that propose different taxonomies for GNNs-based methods {%cite bronstein2017geometric hamilton2017representation battaglia2018relational lee2018graph wu2020comprehensive %}. In this blog post, rather than presenting an exhaustive survey of GNNs, our goal is to define the building blocks including definitions and computational schemes. Besides, we focus on applications in RL and present a short overview of standard methods.  

### Relational Inductive Bias with Graph Neural Networks

First, we propose a definition for the central component of GNNs: the graph. 

**Graph.** A graph is a mathematical structure used to model _pairwise relations_ between _objects_. More formally, we denote a graph by an ordered pair $$G=(V, E)$$, where $$V$$ is the set of vertices or nodes---the objects---and $$E$$ is the set of edges---the pairwise relations. We denote a single node by $$v_i \in V$$, and an edge traveling from node $$v_i$$ to node $$v_j$$ as $$e_{ij} \in E$$. We also define the neighborhood of a node $$v_i$$ to be the set of nodes to which $$v_i$$ is connected by an edge. Formally, this set is defined as 

$$
    \mathcal{N}(v_i) = \{v_j \in V~|~e_{ij} \in E\}.
$$ 

Finally, we consider some global features which characterize the whole graph, and we denote them by $$u$$.

**Undirected and Directed Graphs.** The definition above suggests that the edges of a graph $$G$$ are inherently directed from a _source_ node to a _recipient_ node. In some special scenarios, a graph can be _undirected_: that is, $$e_{ij} = e_{ji}$$ for each pair of nodes $$v_i$$ and $$v_j$$. In this case, the relation between nodes is said to be _symmetric_. If the edges are distinguished from their inverted counterparts ($$e_{ij} \neq e_{ji}$$), then the graph is said to be _directed_.

#### Graph Input

The input of a graph corresponds to the parsed input features of all its nodes, all its edges and some other global features characterizing the whole system. Active lines of research that are orthogonal to our work are exploring methods that enable the extraction of such parsed features from raw sensory data {%cite watters2017visual van2018relational li2018learning kipf2018neural %}. To simplify our study, we suppose the existence of a _predefined feature extractor_ that automatically generates input values for each node and edge. For simplicity, we respectively denote the input features of node $$i$$, edge $$i \rightarrow j$$ and global features by $$v_i$$, $$e_{ij}$$ and $$u$$.

#### Graph Output

Depending on the graph structure and the task at hand, the output of the graph can focus on different graph levels. If the functions used to produce this output are modeled by neural networks, then we speak about GNNs. 

__Node-level.__ This level focuses on the nodes of the graph. In this scenario, input features including node, edge and global features are used to produce a new embedding for each node. This can be used to perform regression and classification at the level of nodes and learn about the physical dynamics of each object {%cite battaglia2016interaction chang2016compositional wang2018nervenet sanchez2018graph %}.

__Edge-level.__ This level focuses on the edges of the graph. The output of the computational scheme in this case are the updated features of each node after propagating the information between all the nodes. For instance, it can be used to make decisions about interactions among the different objects {%cite kipf2018neural hamrick2018relational %}.

__Graph-level.__ This level focuses on the entire graph. The output corresponds to a global embedding computed after propagating the information between all nodes of the graph. It can be used by embodied agents to produce actions in multi-object scenarios {%cite akakzia2021grounding akakzia2022learning %}, to answer questions about a visual scene {%cite santoro2017simple %} or to extract the global properties molecules in chemistry {%cite gilmer2017neural %}.

#### Graph Computation

So far, we have formally defined graphs and distinguished three types of attention-levels which define their output. Thereafter, we explain how exactly the computation of this output is conducted. The computational scheme within GNNs involves two main properties. First, it is based on _shared_ neural networks which are used to compute the updated features of all the nodes and edges. Second, it uses _aggregation functions_ that pool these features in order to produce the output. These two properties provide GNNs with good combinatorial generalization capabilities. In fact, not only it enables good transfer between different nodes and edges (based on the shared networks), but also it leverages permutation invariance (based on the aggregation scheme).

We denote the shared neural networks between the nodes by $$NN_{nodes}$$, the shared neural networks between edges by $$NN_{edges}$$, and the readout neural network that produces the global output of the GNN by $$NN_{readout}$$. Besides, we focus on _graph-level output_. The full computational scheme is based on three steps: the _edge updates_, the _node updates_ and the _graph readout_.

__The edge update step.__ The edge update step consists in using the input features involving each edge $$i \rightarrow j$$ to compute its updated features, which we note $$e'_{ij}$$. More precisely, we consider the global input feature $$u$$, the input features of the source node $$v_i$$ and the input features of the recipient node $$v_j$$. We use the shared network $$NN_{edges}$$ to compute the updated features of all the edges. Formally, the updated features $$e'_{ij}$$ of the edge $$i \rightarrow j$$ are computed as follows: 

$$

    e'_{ij}~=~NN_{edges}(v_i, v_j, e_{ij}, u).

$$

__The node update step.__ The node update step aims at computing the updated features of all the nodes. We note $$v'_{i}$$ these updated features for node $$i$$. To do so, the input features of the underlying node, the global features as well as the aggregation of the updated features of the incoming edges to $$i$$ are considered. The incoming edges to $$i$$ correspond to edges whose source nodes are necessarily in the neighborhood of $$i$$, $$\mathcal{N}(i)$$. The shared network $$NN_{nodes}$$ is used in this computation. Formally, the updated features $$v'_{i}$$ of the node $$i$$ are obtained as follows:

$$

    v'_{i}~=~NN_{nodes}(v_i, Agg_{i \in \mathcal{N}(i)}(e'_{ij}), u).

$$
 
__The graph readout step.__ The graph readout step computes the global output of the graph. This quantity is obtained by aggregating all the updated features of the nodes within the graph. It uses the readout neural network $$NN_{readout}$$. Formally, the output $$o$$ of the GNN is computed as follows:

$$

    o~=~NN_{readout}(Agg_{i \in graph}(v'_{i})).

$$

The computational steps we described above can be used in some other order. For example, one can first perform the node update using the input features of edges, then perform the edge updates using the updated nodes features. This choice usually depends on the domain and task at hand. Besides, our descriptions above are categorized within the family of _convolutional GNNs_ {%cite bruna2013spectral henaff2015deep defferrard2016convolutional kipf2016semi levie2018cayleynets gilmer2017neural akakzia2022learning %}, which generalize the operation of convolution from grid data to graph data by pooling features of neighbors when updating each node. There exist other categories of GNNs, such as _graph auto-encoders_ {%cite cao2016deep wang2016structural kipf2016variational pan2018adversarially li2018learning %}, _spatio-temporal GNNs_ {%cite yu2017spatio li2017diffusion seo2018structured guo2019attention %} and _recurrent GNNs_ {%cite scarselli2005graph gallicchio2010graph li2015gated dai2018learning %}. Finally, the aggregation module used to perform node-wise pooling can be either some predefined permutation-invariant function such as sum, max or mean, or a more sophisticated self-attention-based function that learns attention weights for each node {%cite velivckovic2017graph %}.

### Overview on Graph Neural Networks in RL

Recently, Graph Neural Networks have been widely used in Reinforcement Learning. In fact, they promote sample efficiency, especially in multi-object manipulation domains, where object invariance becomes crucial for generalization. In this paragraph, we introduce an overview over recent works in RL using GNNs. We divide the works in two categories: GNNs used for _model-based_ RL and for _model-free_ RL. 

**Model-based Reinforcement Learning.** The idea of using GNNs in model-based reinforcement learning settings mainly amounts to representing the perceived world of the artificial agents with graphs. Recent papers have been using GNNs to learn prediction models by construction graph representations using the bodies and joints of the agents {%cite wang2016learning hamrick2017metacontrol sanchez2018graph %}. This approach is shown to be successful in prediction, system identification and planning. However, these approaches struggle when the structure of the components and joints of the agent are different. For example, they work better on the Swimmer environment than HalfCheetah, since the latter contains more joints corresponding to different components (back leg, front leg, head ...). Other approaches use Interaction Networks {%cite battaglia2016interaction %}, which are a particular type of GNNs to implement transition models of the environment which they later use for imagination-based optimization {%cite hamrick2017metacontrol %} or planning from scratch {%cite wang2016learning %}

**Model-free Reinforcement Learning.** GNNs are also used in model-free reinforcement learning to model the policy and / or the value function {%cite wang2018nervenet zambaldi2018relational bapst2019structured li2019towards colas2020language akakzia2021grounding %}. On the one hand, like the model-based setting, some approaches use them to represent the agent's body and joints as a graph where the different components interact with each other to produce an action {%cite wang2018nervenet %}. On the other hand, other approaches use it to represent the world in term of separate entities and attempt to capture the relational features between them {%cite zambaldi2018relational bapst2019structured li2019towards colas2020language akakzia2021grounding %}.


### Limitations 

In spite of their generalization capacities provided by their permutation invariance, GNNs still show some limitations to solve some classes of problems such as discriminating between certain non-isomorphic graphs {%cite kondor2018generalization %}. Moreover, notions like recursion, control flow and conditional iteration are not straightforward to represent with graphs, and might require some domain-specific tweaks (for example, in interpreting abstract syntax trees). In fact, symbolic  programs using probabilistic models are shown to work better on these classes of problems {%cite tenenbaum2011grow goodman2014concepts lake2015human %}. But more importantly, a more pressing question is about the origin of the graph networks that most of the methods work on. In fact, most approaches that use GNNs use graphs with predefined entities corresponding to structured objects. Removing this assumption, it is still unclear how to convert sensory data into more structured graph-like representations. Some lines of active research are exploring these issues {%cite watters2017visual van2018relational li2018learning kipf2018neural %}.

<!--- References -->

{% bibliography --cited %}