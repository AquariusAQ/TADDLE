## A APPENDIX

#### A.1 PROOF OF PROPOSITION 3.5

Proof. Since $\Omega$ is a compact metric space, $C(\Omega)$ is separable space. Let $\{ \mu_{n} \}$ be a bounded sequence in $\mathcal{M}(\Omega) \cong \mathcal{C}(\Omega)^{*}$. Then, by the separable version of the Banach-Alaoglu Theorem, there exists a weak* convergent subsequence $\{ \mu_{n_{k}} \}$ such that $\mu_{n_{k}} \xrightarrow{w^{*}} \mu$ (see Problem 10 of Chapter 4.9 in Kreyszig (1991)). Define $\Gamma := \{\sigma(x,\cdot) \in C(\Omega) : x \in \mathcal{X}\}$. Since $\Gamma$ is uniformly bounded

and pointwise equicontinuous, we have the following (see Exercise 8.10.134 in Bogachev & Ruas (2007)):

 $$ \begin{align*}\lim_{n\to\infty}\left\|A\mu_{n_{k}}-A\mu\right\|_{C(\mathcal{X})}&=\lim_{n\to\infty}\sup_{x\in\mathcal{X}}\left|\int\sigma(x,w)d(\mu_{n_{k}}-\mu)(w)\right|\\&=\lim_{n\to\infty}\sup_{f\in\Gamma}\left|\int fd(\mu_{n_{k}}-\mu)\right|=0.\end{align*} $$ 

#### A.2 PROOF OF PROPOSITION 3.7

Proof. Let $\bigoplus_{i\in I}^{p}\mathcal{B}_{i}$ be a feature space and define a feature map $\mathbf{s}:\mathcal{X}\to\left(\bigoplus_{i\in I}^{p}\mathcal{B}_{i}\right)^{*}$ as $\mathbf{s}(x)=\Phi((ev_{x}^{i})_{i\in I})$ for $x\in\mathcal{X}$, where $\Phi:\bigoplus_{i\in I}^{q}\mathcal{B}_{i}^{*}\to\left(\bigoplus_{i\in I}^{p}\mathcal{B}_{i}\right)^{*}$ is the isometric isomorphism defined in equation 2.1. Now, there is a linear transformation $\mathcal{S}:\bigoplus_{i\in I}^{p}\mathcal{B}_{i}\to\mathbb{R}^{\mathcal{X}}$ by $(\mathcal{S}(f_{i})_{i\in I})(x)=\langle\mathbf{s}(x),(f_{i})_{i\in I}\rangle$ for $(f_{i})_{i\in I}\in\bigoplus_{i\in I}^{p}\mathcal{B}_{i}$ and $x\in\mathcal{X}$. Then, by the Theorem 3.3, $\bigoplus_{i\in I}^{p}\mathcal{B}_{i}/\ker(\mathcal{S})=\operatorname{im}(\mathcal{S})$ is an RKBS on $\mathcal{X}$ with the norm $\|f\|_{\mathcal{B}}=\inf_{(f_{i})_{i\in I}\in\mathcal{S}^{-1}(f)}\|(f_{i})_{i\in I}\|_{\bigoplus_{i\in I}^{p}\mathcal{B}_{i}}$.

#### A.3 Proof of Lemma 4.1

Proof. We know that for each  $ i \in I $ ,  $ \pi_i $  is a surjective bounded linear operator, and its norm satisfies  $ \|\pi_i\| \leq 1 $  (see III §4 Theorem 4.2 in Conway (1997)). Additionally, there is an unique linear map  $ (\pi_i)_{i \in I} : \prod_{i \in I} X_i \to \prod_{i \in I} X_i / D_i $  such that  $ \pi_j \circ p_j = q_j \circ (\pi_i)_{i \in I} $  for all  $ j \in I $ , where  $ p_j $  and  $ q_j $  are j-th canonical projections of  $ \prod_{i \in I} X_i $  and  $ \prod_{i \in I} X_i / D_i $ , respectively. Consider the restriction of  $ (\pi_i)_{i \in I} $  to  $ \bigoplus_{i \in I}^p X_i $  and denote it by  $ \widetilde{(\pi_i)_{i \in I}} : \bigoplus_{i \in I}^p X_i \to \prod_{i \in I} X_i / D_i $ . Let  $ (x_i)_{i \in I} \in \bigoplus_{i \in I}^p X_i $ . Since  $ (\pi_i)_{i \in I} = (x_i)_{i \in I} = (\pi_i(x_i))_{i \in I} \in \prod_{i \in I} X_i / D_i $  and  $ \sum_{i \in I} \|\pi_i(x_i)\|^p_{X_i / D_i} \leq \sum_{i \in I} \|x_i\|^{p}_{X_i} < \infty $ , it follows that  $ \text{im} \left( (\pi_i)_{i \in I} \right) \subset \bigoplus_{i \in I}^p X_i / D_i $ . From this, we also know that  $ (\widetilde{\pi_i})_{i \in I} $  is a bounded operator with norm less than 1.

It remains to show the surjectivity of  $ (\widetilde{\pi_{i}})_{i\in I}:\bigoplus_{i\in I}^{p}X_{i}\rightarrow\bigoplus_{i\in I}^{p}X_{i}/D_{i} $ . Let  $ (\pi_{i}(x_{i}))_{i\in I}\in\bigoplus_{i\in I}^{p}X_{i}/D_{i} $ . Then, we have  $ \sum_{i\in I}\inf_{d_{i}\in D_{i}}\|x_{i}+d_{i}\|_{X_{i}}^{p}=\sum_{i\in I}(\inf_{d_{i}\in D_{i}}\|x_{i}+d_{i}\|_{X_{i}})^{p}=\sum_{i\in I}\|\pi_{i}(x_{i})\|_{X_{i}/D_{i}}^{p}<\infty $  and the set  $ N=\{i\in I:\|\pi_{i}(x_{i})\|_{X_{i}/D_{i}}>0\} $  is countable. Let  $ f:\mathbb{N}\to N $  be a reordering bijection. From the definition of the infimum, for each  $ k\inN $ , we can take  $ \tilde{d}_{f(k)}\in D_{f(k)} $  such that

 $$ \|x_{f(k)}+\tilde{d}_{f(k)}\|_{X_{f(k)}}^{p}<\inf_{d_{f(k)}\in D_{f(k)}}\|x_{f(k)}+d_{f(k)}\|_{X_{f(k)}}^{p}+\frac{1}{k^{2}}. $$ 

Then, we have that:

 $$ \begin{align*}\sum_{i\in N}\|x_{i}+\tilde{d}_{i}\|^{p}_{X_{i}}&=\sum_{k=1}^{\infty}\|x_{f(k)}+\tilde{d}_{f(k)}\|^{p}_{X_{f(k)}}\\&<\sum_{k=1}^{\infty}\inf_{d_{f(k)}\in D_{f(k)}}\|x_{f(k)}+d_{f(k)}\|^{p}_{X_{f(k)}}+\sum_{k=1}^{\infty}\frac{1}{k^{2}}<\infty.\end{align*} $$ 

Thus, if we take  $ x_{i}^{\prime}=\begin{cases}x_{i}+\tilde{d}_{i}&if i\in N,\\0&if i\in I\setminus N\end{cases} $ , then  $ (x_{i}^{\prime})_{i\in I}\in\bigoplus_{i\in I}^{p}X_{i} $  and  $ (\widetilde{\pi_{i}})_{i\in I}((x_{i}^{\prime})_{i\in I})=(\pi_{i}(x_{i}))_{i\in I} $ . We can also prove the (2) directly.

#### A.4 PROOF OF PROPOSITION 4.2

Proof. From the Lemma 4.1, we know that there is a surjective bounded linear operator $\widetilde{(\pi_{i})_{i\in I}}$ :

$\bigoplus_{i\in I}^{p}\Psi_{i}\rightarrow\bigoplus_{i\in I}^{p}\Psi_{i}/\ker A_{i}$ and an isometric isomorphism $\widetilde{(\hat{A}_{i})_{i\in I}}:\bigoplus_{i\in I}^{p}\Psi_{i}/\ker A_{i}\rightarrow$

 $ \bigoplus_{i\in I}^{p}B_{i} $ . Let  $ \Phi:\bigoplus_{i\in I}^{q}\mathcal{B}_{i}^{*}\to\big(\bigoplus_{i\in I}^{p}\mathcal{B}_{i}\big)^{*} $  be the isometric isomorphism defined in equation 2.1. Since  $ (ev_{x}^{i})_{i\in I}\in\bigoplus_{i\in I}^{q}\mathcal{B}_{i}^{*} $  for all  $ x\inX $ , we can apply the Proposition 3.7 to deduce that there is an RKBS triple for the summation of RKBSs  $ \sum_{i\in I}^{p}\mathcal{B}_{i}=(\bigoplus_{i\in I}^{p}\mathcal{B}_{i},\mathbf{s},\mathcal{S}) $ . Consider the map  $ A:=\mathcal{S}\circ(\widetilde{\hat{A}_{i}})_{i\in I}\circ(\widetilde{\pi_{i}})_{i\in I}=\mathcal{S}\circ(\widetilde{A_{i}})_{i\in I}:\bigoplus_{i\in I}^{p}\Psi_{i}\to\mathbb{R}^{\mathcal{X}} $ . To verify the map A is indeed an RKBS map, we show the following holds

 $$ \left(A(\mu_{i})_{i\in I}\right)(x)=\left(\mathcal{S}\left(\widetilde{(A_{i})_{i\in I}(\mu_{i})_{i\in I}}\right)\right)(x)=\left\langle\Phi((ev_{x}^{i})_{i\in I})\circ\widetilde{(A_{i})_{i\in I}},(\mu_{i})_{i\in I}\right\rangle $$ 

for all $x\in\mathcal{X}$ and $(\mu_{i})_{i\in I}\in\bigoplus_{i\in I}^{p}\Psi_{i}$. Thus, if we define a feature map $\psi:\mathcal{X}\to\big(\bigoplus_{i\in I}^{p}\Psi_{i}\big)^{*}$ by $\psi(x)=\Phi((ev_{x}^{i})_{i\in I})\circ(\widehat{A_{i}})_{i\in I}\in\big(\bigoplus_{i\in I}^{p}\Psi_{i}\big)^{*}$ for $x\in\mathcal{X}$, then we get an RKBS triple $\mathcal{B}=(\bigoplus_{i\in I}^{p}\Psi_{i},\psi,A)$. Since $(\widehat{A_{i}})_{i\in I}\circ(\pi_{i})_{i\in I}$ is surjective, $\mathrm{im}(A)=\mathrm{im}(\mathcal{S})$ in terms of set equality. Also we note that, by the Theorem 3.3, $\mathcal{B}=\mathrm{im}(A)$ and $\sum_{i\in I}^{p}\mathcal{B}_{i}=\mathrm{im}(\mathcal{S})$ as sets. Since $\mathrm{im}(A)$ and $\mathrm{im}(\mathcal{S})$ both inherit the same algebraic structure from $\mathbb{R}^{\mathcal{X}}$, we can deduce that they are the same as vector space. The only remaining part of the proof is to show that for any $f\in\mathcal{B}$, $\|f\|_{\mathcal{B}}=\|f\|_{\sum_{i\in I}^{p}\mathcal{B}_{i}}$.

To prove (2), suppose that we have the RKBS triple $\mathcal{B}=(\bigoplus_{i\in I}^{p}\Psi_{i},\psi,A)$. We denote $\Phi_{0}:\bigoplus_{i\in I}^{q}\Psi_{i}^{*}\to\big(\bigoplus_{i\in I}^{p}\Psi_{i}\big)^{*}$ as the isometric isomorphism defined in equation 2.1. Since $\psi(x)\in\big(\bigoplus_{i\in I}^{p}\Psi_{i}\big)^{*}$ for all $x\in\mathcal{X}$, we know that

 $$ \Phi_{0}^{-1}\left(\psi(x)\right)\in\bigoplus_{i\in I}^{q}\Psi_{i}^{*},\quad\left\|\Phi_{0}^{-1}\left(\psi(x)\right)\right\|_{\bigoplus_{i\in I}^{q}\Psi_{i}^{*}}<\infty. $$ 

Now, we define for each  $ i \in I $ ,  $ \psi_i : X \to \Psi_i^* $  by  $ \psi_i(x) = p_i (\Phi_0^{-1}(\psi(x))) $  for  $ x \in X $ , where  $ p_i $  is i-th canonical projection on  $ \prod_{i \in I} \Psi_i^* $ . Then, for each  $ i \in I $ , there is an RKBS map  $ A_i : \Psi_i \to R^X $  defined by  $ (A_i \mu_i)(x) = \langle \psi_i(x), \mu_i \rangle $  for  $ x \in X $  and  $ \mu_i \in \Psi_i $ . From the Theorem 3.3, we can get a family of RKBS triples  $ \{\mathcal{B}_i = (\Psi_i, \psi_i, A_i)\}_{i \in I} $ . Let  $ \Phi : \bigoplus_{i \in I}^q \mathcal{B}_i^* \to \bigoplus_{i \in I}^p \mathcal{B}_i $ ^* be the isometric isomorphism defined in equation 2.1. By the above equation A.1, we can deduce that  $ (\psi_i(x))_{i \in I} \in \bigoplus_{i \in I}^q \Psi_i^* $ . Thus, the Remark 3.8 implies the existence of an RKBS triple for the sum of RKBSs  $ \sum_{i \in I}^p \mathcal{B}_i = (\bigoplus_{i \in I}^p \mathcal{B}_i, s, S) $ . From the following series of equations, we can see that  $ A = S \circ (\hat{A}_i)_{i \in I} \circ (\pi_i)_{i \in I} $ . For  $ x \in X $  and  $ (\mu_i)_{i \in I} \in \bigoplus_{i \in I}^p \Psi_i $ , we have that

 $$ \begin{align*}\left(A\left((\mu_{i})_{i\in I}\right)\right)(x)&=\langle\psi(x),(\mu_{i})_{i\in I}\rangle=\left\langle\Phi_{0}\left(\Phi_{0}^{-1}(\psi(x))\right),(\mu_{i})_{i\in I}\right\rangle=\sum_{i\in I}\left\langle p_{i}(\Phi_{0}^{-1}(\psi(x))),\mu_{i}\right\rangle\\&=\sum_{i\in I}\langle\psi_{i}(x),\mu_{i}\rangle=\left\langle\Phi((ev_{x}^{i})_{i\in I}),(A_{i}\mu_{i})_{i\in I}\right\rangle=(\mathcal{S}\left((A_{i}\mu_{i})_{i\in I}\right))(x)\\&=\left(\widetilde{\mathcal{S}\left((\widetilde{A_{i}})_{i\in I}((\mu_{i})_{i\in I})\right)}\right)(x)=\left(\left(\widetilde{\mathcal{S}\circ(\widetilde{A_{i}})_{i\in I}\circ(\widetilde{\pi_{i}})_{i\in I}}\right)((\mu_{i})_{i\in I})\right)(x).\end{align*} $$ 

For similar reasons as the previous case, we only need to prove that for any  $ f \in B $ ,  $ \|f\|_{B} = \|f\|_{\sum_{i \in I}^{p} B_{i}} $ .

We start by exploring the definition of each norm. The norm on the RKBS  $ \sum_{i\in I}^{p}B_{i} $  is given by

 $$ \begin{align*}\|f\|_{\sum_{i\in I}^{p}\mathcal{B}_{i}}^{p}&=\inf\left\{\|(f_{i})_{i\in I}\|_{\bigoplus_{i\in I}^{p}\mathcal{B}_{i}}^{p}:(f_{i})_{i\in I}\in\mathcal{S}^{-1}(f)\right\}\\&=\inf\left\{\sum_{i\in I}\inf\left\{\|\mu_{i}\|_{\Psi_{i}}^{p}:\mu_{i}\in A_{i}^{-1}(f_{i})\right\}:(f_{i})_{i\in I}\in\mathcal{S}^{-1}(f)\right\}\end{align*} $$ 

for  $ f \in \sum_{i \in I}^{p} B_{i} $ . The norm on the RKBS B is given by

 $$ \begin{aligned}\|f\|_{\mathcal{B}}^{p}&=\inf\left\{\|(\mu_{i})_{i\in I}\|_{\bigoplus_{i\in I}^{p}\Psi_{i}}^{p}:(\mu_{i})_{i\in I}\in A^{-1}(f)\right\}=\inf\left\|A^{-1}(f)\right\|_{\bigoplus_{i\in I}^{p}\Psi_{i}}^{p}\\&=\inf\left\|\widetilde{(A_{i})_{i\in I}}^{-1}\circ\mathcal{S}^{-1}(f)\right\|_{\bigoplus_{i\in I}^{p}\Psi_{i}}^{p}=\inf\left\|\bigcup_{(f_{i})_{i\in I}\in\mathcal{S}^{-1}(f)}\widetilde{(A_{i})_{i\in I}}^{-1}((f_{i})_{i\in I})\right\|_{\bigoplus_{i\in I}^{p}\Psi_{i}}^{p}\\&=\inf\bigcup_{(f_{i})_{i\in I}\in\mathcal{S}^{-1}(f)}\left\|\widetilde{(A_{i})_{i\in I}}^{-1}((f_{i})_{i\in I})\right\|_{\bigoplus_{i\in I}^{p}\Psi_{i}}^{p}\\&=\inf\left\{\inf\left\|\widetilde{(A_{i})_{i\in I}}^{-1}((f_{i})_{i\in I})\right\|_{\bigoplus_{i\in I}^{p}\Psi_{i}}^{p}:(\boldsymbol{f}_{i})_{i\in I}\in\mathcal{S}^{-1}(f)\right\}\\&=\inf\left\{\inf\left\{\sum_{i\in I}\|\mu_{i}\|_{\Psi_{i}}^{p}:(\mu_{i})_{i\in I}\in\widetilde{(A_{i})_{i\in I}}^{-1}((f_{i})_{i\in I})\right\}:(\boldsymbol{f}_{i})_{i\in I}\in\mathcal{S}^{-1}(f)\right\}\end{aligned} $$ 

for $f \in \mathcal{B}$. If we denote the set $\left\{\sum_{i \in I} \|\mu_i\|_{\Psi_i}^p : (\mu_i)_{i \in I} \in (\widetilde{A_i})_{i \in I}^{-1} ((f_i)_{i \in I})\right\}$ by $\mathcal{C}$, then we conclude the proof by showing that:

 $$ \sum_{i\in I}\inf\left\{\|\mu_{i}\|_{\Psi_{i}}^{p}:\mu_{i}\in A_{i}^{-1}(f_{i})\right\}=\inf\mathcal{C} $$ 

for all $(f_{i})_{i\in I}\in\mathcal{S}^{-1}(f)$. To show that $\sum_{i\in I}\inf\left\{\|\mu_{i}\|_{\Psi_{i}}^{p}:\mu_{i}\in A_{i}^{-1}(f_{i})\right\}$ is a lower bound for $\mathcal{C}$, we note that $(\mu_{i})_{i\in I}\in\widetilde{(A_{i})_{i\in I}}^{-1}((f_{i})_{i\in I})$ is equivalent to

 $$ (\mu_{i})_{i\in I}\in\bigoplus_{i\in I}^{p}\Psi_{i}\mathrm{a n d}\forall i\in I,A_{i}\mu_{i}=f_{i}. $$ 

Let (νi)i∈I ∈ (A i)i∈I−1 ((f i)i∈I).

Then, by the equation A.2, we have that  $ \inf \left\{\|\mu_{i}\|_{\Psi_{i}}^{p} : \mu_{i} \in A_{i}^{-1}(f_{i})\right\} \leq \|\nu_{i}\|_{\Psi_{i}}^{p} $  for all  $ i \in I $ . Thus, we deduce that  $ \sum_{i \in I} \inf \left\{\|\mu_{i}\|_{\Psi_{i}}^{p} : \mu_{i} \in A_{i}^{-1}(f_{i})\right\} \leq \sum_{i \in I} \|\nu_{i}\|_{\Psi_{i}}^{p} $ . Now, we have to show that  $ \sum_{i \in I} \inf \left\{\|\mu_{i}\|_{\Psi_{i}}^{p} : \mu_{i} \in A_{i}^{-1}(f_{i})\right\} $  is the greatest lower bound of C. Let c be an any lower bound of the set C. Since we already assumed that  $ (f_{i})_{i \in I} \in \mathcal{S}^{-1}(f) $ , the norm of  $ (f_{i})_{i \in I} $  in  $ \bigoplus_{i \in I}^{p} B_{i} $  is finite. That is, we know that  $ \sum_{i \in I} \inf \left\{\|\mu_{i}\|_{\Psi_{i}}^{p} : \mu_{i} \in A_{i}^{-1}(f_{i})\right\} = \sum_{i \in I} \|f_{i}\|_{\mathcal{B}_{i}} < \infty $ . We denote the set  $ \{i \in I : \inf \left\{\|\mu_{i}\|_{\Psi_{i}}^{p} : \mu_{i} \in A_{i}^{-1}(f_{i})\right\} \neq 0\} $  by H. Then, for  $ i \in I \setminus H $ ,  $ \inf \left\{\|\mu_{i}\|_{\Psi_{i}}^{p} : \mu_{i} \in A_{i}^{-1}(f_{i})\right\} = 0 $ . Hence, there is a sequence  $ \{\nu_{i}^{n}\}_{n \in N} \in A_{i}^{-1}(f_{i}) $ , so that  $ \|\nu_{i}^{n} - 0\|_{\Psi_{i}}^{p} \to 0 $  as  $ n \to \infty $ . Furthermore, since  $ A_{i}^{-1}(f_{i}) $  is a translation of ker  $ A_{i} $ , by the equation 3.1,  $ A_{i}^{-1}(f_{i}) $  is a closed subset in  $ \Psi_{i} $ . Therefore, we deduce that

 $$ 0\in A_{i}^{-1}(f_{i})for all i\in I\setminus H $$ 

For the case of H, note that H is a countable subset of I. Accordingly, we may take a reordering bijection $g: \mathbb{N} \to H$. By simply using the definition of the infimum, for any $1 > \epsilon > 0$ and for any $g(n) \in H$, there is a $\nu_{g(n)} \in A_{g(n)}^{-1}(f_{g(n}})$ such that

 $$ \inf\left\{\|\mu_{g(n)}\|_{\Psi_{g(n)}}^{p}:\mu_{g(n)}\in A_{g(n)}^{-1}(f_{g(n)})\right\}+\frac{1}{4}\cdot\frac{1}{2^{n}}\cdot\epsilon>\|\nu_{g(n)}\|_{\Psi_{g(n)}}^{p}. $$ 

Combining the above results, we obtain the following:

 $$ \sum_{i\in I}\inf\left\{\|\mu_{i}\|_{\Psi_{i}}^{p}:\mu_{i}\in A_{i}^{-1}(f_{i})\right\}+\epsilon $$ 

 $$ >\sum_{i\in I}\inf\left\{\|\mu_{i}\|_{\Psi_{i}}^{p}:\mu_{i}\in A_{i}^{-1}(f_{i})\right\}+\sum_{n=1}^{\infty}\frac{1}{4}\cdot\frac{1}{2^{n}}\cdot\epsilon $$ 

 $$ =\sum_{i\in H}\inf\left\{\|\mu_{i}\|_{\Psi_{i}}^{p}:\mu_{i}\in A_{i}^{-1}(f_{i})\right\}+\sum_{n=1}^{\infty}\frac{1}{4}\cdot\frac{1}{2^{n}}\cdot\epsilon $$ 

 $$ =\sum_{n=1}^{\infty}\left(\inf\left\{\|\mu_{g(n)}\|_{\Psi_{g(n)}}^{p}:\mu_{g(n)}\in A_{g(n)}^{-1}(f_{g(n)})\right\}+\frac{1}{4}\cdot\frac{1}{2^{n}}\cdot\epsilon\right) $$ 

 $$ >\sum_{n=1}^{\infty}\|\nu_{g(n)}\|_{\Psi_{g(n)}}^{p}=\sum_{i\in H}\|\nu_{i}\|_{\Psi_{i}}^{p}. $$ 

Define  $ \xi_{i}=\begin{cases}\nu_{i}&if i\in H,\\0&if i\in I\setminus H.\end{cases} $ . Then, by the equation A.3 and equation A.4, we know that for all  $ i\in I $ ,  $ \xi_{i}\in A_{i}^{-1}(f_{i}) $ . In addition, from the inequalities in equation A.9, we also know that  $ \sum_{i\in I}\|\xi_{i}\|_{\Psi_{i}}^{p}\leq\sum_{i\in I}\|f_{i}\|_{\mathcal{B}_{i}}^{p}+1<\infty $ . Thus, by the equation A.2, we deduce that  $ (\xi_{i})_{i\in I}\in(\widetilde{A_{i}})_{i\in I}\stackrel{-1}{((f_{i})_{i\in I})} $  (i.e.,  $ \sum_{i\in I}\|\xi_{i}\|_{\Psi_{i}}^{p}\in\mathcal{C} $ ). Finally, the following show that  $ \sum_{i\in I}\inf\left\{\|\mu_{i}\|_{\Psi_{i}}^{p}:\mu_{i}\in A_{i}^{-1}(f_{i})\right\} $  is the greatest lower bound of C:

 $$ \sum_{i\in I}\inf\left\{\|\mu_{i}\|_{\Psi_{i}}^{p}:\mu_{i}\in A_{i}^{-1}(f_{i})\right\}+\epsilon>\sum_{i\in H}\|\nu_{i}\|_{\Psi_{i}}^{p}=\sum_{i\in I}\|\xi_{i}\|_{\Psi_{i}}^{p}\geq\mathbf{c}\quad for all1>\epsilon>0, $$ 

where c is a lower bound of the set C.

#### A.5 Proof of Proposition 4.3

Proof. Define a feature map  $ \psi_{1}: \mathcal{X} \to \Psi_{1}^{*} $  by  $ \psi_{1}(x) = \psi_{2}(x) \circ \xi $  for  $ x \in X $  and a linear map  $ A_{1}: \Psi_{1} \to R^{X} $  by  $ (A_{1}\mu)(x) = <\psi_{1}(x), \mu > $  for  $ x \in X $  and  $ \mu \in \Psi_{1} $ . Then, we deduce that  $ A_{1} = A_{2} \circ \xi $ . Furthermore,  $ \mathcal{B}_{1} = (\Psi_{1}, \psi_{1}, A_{1}) $  is an RKBS. Consider the map  $ \overline{\xi}: \Psi_{1}/\ker A_{2} \circ \xi \to \Psi_{2}/\ker A_{2} $  defined by  $ \overline{\xi}([\mu]) = [\xi(\mu)] $  for  $ [\mu] \in \Psi_{1}/\ker A_{2} \circ \xi $ . Since  $ \ker (A_{2} \circ \xi) = \xi^{-1}(\ker A_{2}) $ ,  $ \overline{\xi} $  is a well-defined vector space monomorphism. The remaining proof for establishing surjectivity and isometry is straightforward.

#### A.6 Proof of Theorem 4.4

Proof. By the Definition 3.4, there is a map $\psi: X \to M(\Omega)^{*}$ defined by $\psi(x) = \Lambda^{*}(\iota(\sigma(x,\cdot)))$ for $x \in X$. And there is an RKBS map $A: M(\Omega) \to \mathbb{R}^{X}$ defined by $(A(\mu))(x) = <\psi(x), \mu >$ for all $x \in X$ and $\mu \in M(\Omega)$ such that

 $$ \mathcal{F}_{\sigma}(\mathcal{X},\Omega)\cong_{B}M(\Omega)/\ker A. $$ 

Let  $ \Theta : \bigoplus_{i \in I}^{1} L^{1}(\mu_{i}) \to M(\Omega) $  be the isometric isomorphism defined in equation 2.4. Define a map  $ \overline{\psi} : X \to \left(\bigoplus_{i \in I}^{1} L^{1}(\mu_{i})\right)^{*} $  by  $ \overline{\psi}(x) = \psi(x) \circ \Theta $ . And consider a map  $ \overline{A} : \bigoplus_{i \in I}^{1} L^{1}(\mu_{i}) \to R^{X} $  defined by  $ \overline{A} = A \circ \Theta $ . Then, by the Lemma 4.3, we have that

 $$ M(\Omega)/\ker A\cong\bigoplus_{\mathcal{B}}^{1}L^{1}(\mu_{i})/\ker\overline{A}. $$ 

Now, let $\Phi_{0}:\bigoplus_{i\in I}^{\infty}\left(L^{1}(\mu_{i})\right)^{*}\to\left(\bigoplus_{i\in I}^{1}L^{1}(\mu_{i})\right)^{*}$ be the isometric isomorphism defined in equation 2.1. For each $i\in I$, if we define a map $\overline{\psi}_{i}:X\to\left(L^{1}(\mu_{i})\right)^{*}$ by $\overline{\psi}_{i}(x)=p_{i}\left(\Phi_{0}^{-1}\left(\overline{\psi}(x)\right)\right)$

for $x\in X$ and define a map $\overline{A_{i}}:L^{1}(\mu_{i})\to\mathbb{R}^{X}$ by $(\overline{A_{i}}(h))(x)=\left\langle\overline{\psi}_{i}(x),h\right\rangle$ for $x\in X$ and $h\in L^{1}(\mu_{i})$, then by the Proposition 4.2, we can deduce that

 $$ \bigoplus_{i\in I}^{1}L^{1}(\mu_{i})/\ker\overline{A}\cong\sum_{i\in I}^{1}\mathcal{B}_{i}, $$ 

where $\mathcal{B}_{i}=(L^{1}(\mu_{i}),\overline{\psi}_{i},\overline{A}_{i})$ for all $i\in I$. We want to show that $\mathcal{B}_{i}$ is indeed $\mathcal{L}_{\sigma}(\mu_{i})$ for all $i\in I$. Suppose for each $i\in I$, $\Xi^{i}:L^{\infty}(\mu_{i})\to\left(L^{1}(\mu_{i})\right)^{*}$ is the isometric isomorphism introduced in equation 2.3. According to the Definition 3.6, it suffices to verify that $\overline{\psi}_{i}(x)=\Xi^{i}(\sigma(x,\cdot))$ for all $x\in X$ and $i\in I$. This condition is equivalent to $\overline{\psi}(x)=\Phi_{0}\left((\Xi^{i}(\sigma(x,\cdot)))_{i\in I}\right)$ for all $x\in X$. Hence, we want to prove the following holds: $(\Lambda^{*}(\iota(\sigma(x,\cdot)))\circ\Theta)((f_{i})_{i\in I})=\Phi_{0}\left((\Xi^{i}(\sigma(x,\cdot)))_{i\in I}\right)((f_{i})_{i\in I})$ for all $x\in X$ and $(f_{i})_{i\in I}\in\bigoplus_{i\in I}^{1}L^{1}(\mu_{i})$. First, for the left-hand side, we have:

 $$ \begin{aligned}&\left(\Lambda^{*}\left(\iota(\sigma(x,\cdot))\right)\circ\Theta\right)\left(\left(f_{i}\right)_{i\in I}\right)=\left\langle\Lambda^{*}\left(\iota(\sigma(x,\cdot))\right),\mathcal{M}(K)\sum_{i\in I}\rho_{i}\right\rangle\\&=\left\langle\iota(\sigma(x,\cdot))\circ\Lambda,\mathcal{M}(K)\sum_{i\in I}\rho_{i}\right\rangle=\sum_{i\in I}\left\langle\iota(\sigma(x,\cdot))\circ\Lambda,\rho_{i}\right\rangle\\&=\sum_{i\in I}\left\langle\iota(\sigma(x,\cdot)),\Lambda(\rho_{i})\right\rangle=\sum_{i\in I}\left\langle\Lambda(\rho_{i}),\sigma(x,\cdot)\right\rangle\\&=\sum_{i\in I}\int_{\Omega}\sigma(x,w)d\rho_{i}(w)=\sum_{i\in I}\int_{\Omega}\sigma(x,w)f_{i}(w)d\mu_{i}(w).\\ \end{aligned} $$ 

Next, for the right-hand side, we have:

 $$ \Phi_{0}\left((\Xi^{i}(\sigma(x,\cdot)))_{i\in I}\right)\left((f_{i})_{i\in I}\right)=\sum_{i\in I}\left\langle\Xi^{i}(\sigma(x,\cdot)),f_{i}\right\rangle=\sum_{i\in I}\int_{\Omega}\sigma(x,w)f_{i}(w)d\mu_{i}(w). $$ 

#### A.7 PROOF OF PROPOSITION 5.1

Proof. As we noted in the Definition 3.6, we can easily show that for a given $\pi\in P(\Omega)$, $\mathcal{L}_{\sigma}^{2}(\pi)\subset\mathcal{L}_{\sigma}^{1}(\pi)$ and $\|f\|_{\mathcal{L}_{\sigma}^{1}(\pi)}\leq\|f\|_{\mathcal{L}_{\sigma}^{2}(\pi)}$ for all $f\in\mathcal{L}_{\sigma}^{2}(\pi)$. Let $\{\mu_{i}\}_{i\in I}$ be a maximal singular family containing $\{ \mu_{i}\}_{i\in[n]}$. Consider the map $\iota:\bigoplus_{i\in[n]}^{2}\mathcal{L}_{\sigma}^{2}(\mu_{i})\to\bigoplus_{i\in I}^{1}\mathcal{L}_{\sigma}^{1}(\mu_{i})$ defined by $\iota(\mathbf{x})(i)=\begin{cases}\mathbf{x}(i),&if i\in[n]\\ 0,&if i\in I\setminus[n]\end{cases}$ for $\mathbf{x}\in\bigoplus_{i\in[n]}^{2}\mathcal{L}_{\sigma}^{2}(\mu_{i})$. Since $\iota(\mathbf{x})(i)=\mathbf{x}(i)\in\mathcal{L}_{\sigma}^{2}(\mu_{i})\subset\mathcal{L}_{\sigma}^{1}(\mu_{i})$ for all $i\in[n]$, we know that $\iota(\mathbf{x})\in\prod_{i\in I}\mathcal{L}_{\sigma}^{1}(\mu_{i})$. Furthermore, by the following inequalities $\sum_{i\in I}\|\iota(\mathbf{x})(i)\|_{\mathcal{L}_{\sigma}^{1}(\mu_{i})}=\sum_{i=1}^{n}\|\mathbf{x}(i)\|_{\mathcal{L}_{\sigma}^{1}(\mu_{i})}\leq\sum_{i=1}^{n}\|\mathbf{x}(i)\|_{\mathcal{L}_{\sigma}^{2}(\mu_{i})}&lt;\infty$, we deduce that $\iota(\mathbf{x})\in\bigoplus_{i\in I}^{1}\mathcal{L}_{\sigma}^{1}(\mu_{i})$. Thus, $\iota$ is well-defined linear map.

By the Remark 3.8, we can define the RKBS linear map for the sum of RKBSs  $ S_{1} : \bigoplus_{i \in I}^{1} \mathcal{L}_{\sigma}^{1}(\mu_{i}) \to \mathbb{R}^{\mathcal{X}} $  by  $ \mathcal{S}_{1}((f_{i})_{i \in I})(x) = \sum_{i \in I} f_{i}(x) $  for  $ x \in X $  and  $ (f_{i})_{i \in I} \in \bigoplus_{i \in I}^{1} \mathcal{L}_{\sigma}^{1}(\mu_{i}) $ . And let  $ S_{2} : \bigoplus_{i \in [n]}^{2} \mathcal{L}_{\sigma}^{2}(\mu_{i}) \to \mathbb{R}^{\mathcal{X}} $  be the RKBS linear map defined by  $ \mathcal{S}_{2}((f_{i})_{i \in [n]})(x) = \sum_{i \in [n]} f_{i}(x) $  for  $ x \in X $  and  $ (f_{i})_{i \in [n]} \in \bigoplus_{i \in [n]}^{2} \mathcal{L}_{\sigma}^{2}(\mu_{i}) $ . Now, consider the map  $ \bar{\iota} : \bigoplus_{i \in [n]}^{2} \mathcal{L}_{\sigma}^{2}(\mu_{i}) / \ker S_{2} \to \bigoplus_{i \in I}^{1} \mathcal{L}_{\sigma}^{1}(\mu_{i}) / \ker S_{1} $  defined by  $ \bar{\iota}([\mathbf{x}]) = [\iota(\mathbf{x})] $  for  $ x \in \bigoplus_{i \in [n]}^{2} \mathcal{L}_{\sigma}^{2}(\mu_{i}) $ . From the following fact

 $$ \begin{align*}\iota^{-1}(\ker\mathcal{S}_{1})&=\left\{\mathbf{x}\in\bigoplus_{i\in[n]}^{2}\mathcal{L}_{\sigma}^{2}(\mu_{i}):\iota(\mathbf{x})\in\ker\mathcal{S}_{1}\right\}\\&=\left\{\mathbf{x}\in\bigoplus_{i\in[n]}^{2}\mathcal{L}_{\sigma}^{2}(\mu_{i}):\sum_{i\in I}\left(\iota(\mathbf{x})(i)\right)(x)=0\mathrm{for all}x\in\mathcal{X}\right\}=\ker\mathcal{S}_{2},\end{align*} $$ 

we deduce that  $ \bar{\iota} $  is well-defined monomorphism. If we consider the map  $ \widetilde{\operatorname{id}} = \hat{S}_{1} \circ \bar{\iota} \circ \hat{S}_{2}^{-1} : \sum_{i \in [n]}^{2} \mathcal{L}_{\sigma}^{2}(\mu_{i}) \to \sum_{i \in I} \mathcal{L}_{\sigma}^{1}(\mu_{i}) $ , then it is indeed the identity map. Thus, we have  $ \sum_{i \in [n]}^{2} \mathcal{L}_{\sigma}^{2}(\mu_{i}) \subset \sum_{i \in I} \mathcal{L}_{\sigma}^{1}(\mu_{i}) $ . Furthermore, by the Remark 4.5, we know that  $ \sum_{i \in I} \mathcal{L}_{\sigma}^{1}(\mu_{i}) = \mathcal{F}_{\sigma}(\mathcal{X}, \Omega) $  as a set equality.

#### A.8 PROOF OF PROPOSITION 5.2

Proof. For fixed  $ i \in [n] $ , define  $ \iota: L^{2}(\Omega, \pi_{i}) \to L^{2}(\Omega \times [0, 1], \pi_{i} \otimes \delta_{i/n}) $  by  $ \iota(h)(w, r) = \begin{cases} h(w) & \text{if } r = \frac{i}{n}, \\ 0 & \text{otherwise} \end{cases} $  for  $ w \in \Omega $  and  $ r \in [0, 1] $  where  $ \delta_{i/n} $  is the Dirac measure centred on i/n in ([0, 1],  $ \mathcal{B}([0, 1]) $ ). Then,  $ \iota(h) $  is measurable with respect to  $ (\Omega \times [0, 1], \mathcal{B}(\Omega \times [0, 1])) $  and  $ \int_{\Omega \times [0, 1]} |\iota(h)(w, r)|^{2} d\pi_{i} \otimes \delta_{i/n} < \infty $ . Thus,  $ \iota $  is well-defined linear map.

Now, define $A: L^{2}(\Omega,\pi_{i})\to\mathbb{R}^{\mathcal{X}}$ by $(Ah)(x)=\int_{\Omega}\sigma_{i}(x,w)h(w)d\pi_{i}$ for $h\in L^{2}(\Omega,\pi_{i})$ and $x\in\mathcal{X}$ and $B:L^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n})\to\mathbb{R}^{\mathcal{X}}$ by $(B\tilde{h})(x)=\int_{\Omega\times[0,1]}\sigma(x,w,r)\tilde{h}(w,r)d\pi_{i}\otimes\delta_{i/n}$ for $\tilde{h}\in L^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n})$ and $x\in\mathcal{X}$, which are the RKBS linear maps introduced in the Definition 3.6. Since we know that

 $$ \begin{align*}\iota^{-1}(\ker B)&=\left\{h\in L^{2}(\Omega,\pi_{i}):\iota(h)\in\ker B\right\}=\left\{h\in L^{2}(\Omega,\pi_{i}):B(\iota(h))=0\right\}\\&=\left\{h\in L^{2}(\Omega,\pi_{i}):\int_{\Omega}\int_{[0,1]}\sigma(x,w,r)\iota(h)(w,r)d\delta_{i/n}d\pi_{i}\right\}\\&=\left\{h\in L^{2}(\Omega,\pi_{i}):\int_{\Omega}\sigma_{i}(x,w)h(w)d\pi_{i}\right\}=\ker A,\end{align*} $$ 

it follows that  $ \bar{\iota}:L^{2}(\Omega,\pi_{i})\to L^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n}) $  defined by  $ \bar{\iota}([h])=[\iota(h)] $  for  $ h\in L^{2}(\Omega,\pi_{i}) $  is well-defined monomorphism.

Consider a map $\tilde{\mathrm{id}}=\hat{B}\circ\bar{\iota}\circ\hat{A}^{-1}:\mathcal{L}_{\sigma_{i}}^{2}(\Omega,\pi_{i})\to\mathcal{L}_{\sigma}^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n})$ and let $Ah\in\mathcal{L}_{\sigma_{i}}^{2}(\Omega,\pi_{i})=\mathrm{im}(A)$. Then, we have $\tilde{\mathrm{id}}(Ah)=\tilde{B}\circ\bar{\iota}([h])=\hat{B}([\iota(h)])=B\iota(h)=Ah$. It means that $\tilde{\mathrm{id}}$ is an identity map. Furthermore, we can deduce that

 $$ \begin{aligned}&\|A h\|_{\mathcal{L}_{\sigma}^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n})}=\|B\iota(h)\|_{\mathcal{L}_{\sigma}^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n})}=\|[\iota(h)]\|_{L_{\sigma}^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n})/\ker B}\\ &=\inf_{\tilde{g}\in\ker B}\|\iota(h)+\tilde{g}\|_{L_{\sigma}^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n})}\leq\inf_{g\in\ker A}\|\iota(h)+\iota(g)\|_{L_{\sigma}^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n})}\\ &=\|A h\|_{\mathcal{L}_{\sigma_{i}}^{2}(\Omega,\pi_{i})}.\\ \end{aligned} $$ 

Thus, for  $ i = 1, \ldots, n $ , we have

 $$ \mathcal{L}_{\sigma_{i}}^{2}(\Omega,\pi_{i})\subset\mathcal{L}_{\sigma}^{2}(\Omega\times[0,1],\pi_{i}\otimes\delta_{i/n}) $$ 

and for all  $ f \in \mathcal{L}_{\sigma}^{2}(\Omega, \pi_{i}) $ ,  $ \|f\|_{\mathcal{L}_{\sigma}^{2}(\Omega \times [0,1], \pi_{i} \otimes \delta_{i/n})} \leq \|f\|_{\mathcal{L}_{\sigma_{i}}^{2}(\Omega, \pi_{i})} $ . From this, we can verify that  $ \sum_{i \in [n]}^{2} \mathcal{L}_{\sigma_{i}}^{2}(\Omega, \pi_{i}) \subset \sum_{i \in [n]}^{2} \mathcal{L}_{\sigma}^{2}(\Omega \times [0,1], \pi_{i} \otimes \delta_{i/n}) $  and since  $ \{\pi_{i} \otimes \delta_{i/n}\}_{i=1}^{n} $  is a singular family in  $ P(\Omega \times [0,1]) $ , by the Proposition 5.1, we conclude that  $ \sum_{i \in [n]}^{2} \mathcal{L}_{\sigma_{i}}^{2}(\Omega, \pi_{i}) \subset \mathcal{F}_{\sigma}(\mathcal{X}, \Omega \times [0,1]) $ .  $ \Box $