---
layout: post
title: "Hybrid Goal Exploration Processes"
description: "A framework for learning goals from two different generation sources."
feature_image: images/hgeps.png
date: 2022-09-21
---

> Autotelic agents—agents that are intrinsically motivated to represent, generate and pursue their own goals—aim at growing their repertoire of skills. This implies that they need not only to discover as many goals as possible, but also to learn to achieve each of these goals. When these agents evolve in environments where they have no clue about which goals they can physically reach in the first place, it becomes challenging to handle the _exploration-exploitation_ dilemma. 

<!--more-->

Actually these agents are usually trained to optimize a fitness measure with reference to the distribution of goals they have physically encountered. Yet this distribution shifts as they discover more and more goals. Consequently, such agents have to make sure they still perform well on what they have already encountered, but continue to grow their distribution of discovered goals. This becomes rapidly problematic in hard exploration environments with bottlenecks and sparse rewarding signals, and agents usually require additional tricks to overcome these exploration obstacles. Interestingly, developmental psychology and education sciences highlight the importance of __guided-play__ in the early skill acquisition of infants {%cite wood1976role vygotsky1978mind tomasello1992social lindblom2002social %}. Human caregivers help toddlers overcome their exploration limitation by assisting them whenever it becomes necessary. Our goal is to introduce a goal exploration framework for autotelic agents that benefit from external signals while keeping their intrinsically motivated characteristics. This chapter introduces preliminary studies in which we introduce _Hybrid Goal Exploration Processes_ (HGEPs), a novel family of algorithms that handle exploration of multiple goals from internal and external sources. We study the impact of having this type of coupled exploration on the learning and diversity of goals in different robotics environments.  

