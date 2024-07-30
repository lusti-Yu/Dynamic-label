DYNAT
===
Overview
---
In this paper, we introduce a novel dynamic label adversarial training algorithm called DYNAT for deep learning defense. For this, we innovate a teacher-student
framework for adversarial knowledge distillation, which is achieved by iteratively generating dynamic labels using the maximum predictive probability vector from the teacher model during the training process. Standard adversarial training methods use static labels (ground truth) from datasets for both inner (adversarial example generation) and outer (adversarial training) optimizations. Whereas our DYNAT method follows a “weak to strong” schedule, where the teacher model learns from clean datasets while concurrently injecting the maximum teacher probability vector into the student model’s adversarial training process. That is, the critical focus lies in progressively guiding a student model via a teacher model. Unlike the conventional method of immediately providing a student model with fully standard answers (ground truth labels from the dataset), our novel methodology incrementally enhances the standard answers as the adversarial training progresses. In contrast to prior works, we use the conventional cross-entropy loss for adversarial training, achieving significant improvements in both clean and robustness accuracy when tested on CIFAR-10 and CIFAR-100 datasets and state- of-the-art adversarial attacks.


Dynamic Label Adversarial Training
---

Proposed dynamic label adversarial training (DYNAT) of deep learning models. DYNAT explicitly gives flexibility on the loss functions for adversarial training, whose dynamic label comes from the teacher model. We train the teacher model/network using a dataset with fixed (ground truth) labels (blue labels, $label_1^{nat}, label_2^{nat}, \ldots$.) and concurrently use the maximum value in the teacher model's logits as dynamic labels (orange labels, $l_1^{nat}, l_2^{nat}, \ldots$) for adversarial training of the student model/network. The teacher model $f^t (x_i^{nat})$ takes a clean image $\(x_i^{nat}\)$ as its input and produces a softmax probability value vector, which is converted into labels $l_i^{nat}$ using a one-hot or winner-takes-all principle that is used by student model for computing its cross-entropy $\mathcal{L}_s(f^s (x_i^{adv}),l_i^{nat})$ between students model's output $f^s (x_i^{adv})$ on adversarial image $\(x_i^{adv}\)$. (The adversarial example generated in each iteration of training using an inner optimization) This cross-entropy loss on dynamic label backpropagates to the student model for its dynamic label adversarial training. The dynamic label strength increases from weak to strong as the training loss $\mathcal{L}_t$ minimizes iteratively.


![Image text](https://github.com/lusti-Yu/aaa/blob/main/outer-op.png)



Inner Optimization: Dynamic Adversarial Examples Generation
---

Our inner optimization framework. The natural images are fed into student and teacher models. Then, we use our strategy to extract dynamic labels from teacher outputs (orange labels). We encourage student outputs and dynamic labels to participate together in adversarial example generation via the cross-entropy function. This generated adversarial example is fed back to the student model for the dynamic label adversarial training. 


![Image text](https://github.com/lusti-Yu/aaa/blob/main/inner-opto.png)


Requirements
---
Python3

Pytorch


Get Started
----

Training :
```
python train_dynat_cifar100.py
```

Testing:
```
python auto_attack_eval1.py
```


