\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{dsfont}
\usepackage{graphicx}
\usepackage[utf8]{inputenc}
\usepackage[usenames,dvipsnames]{xcolor}
\usepackage[most]{tcolorbox}
% \usepackage{soul}
% \usepackage{authblk}
% \usepackage[legalpaper, margin=2in]{geometry}
\usepackage{verbatim}

\usepackage{listings}
\lstset{%
  basicstyle=\footnotesize\ttfamily,
  morecomment=[l][\color{black}]{*},
  morecomment=[l][\color{gray}]{\#},
  morecomment=[l][\color{Mahogany}]{//},
  morecomment=[n][\bfseries]{/*}{*/},
  commentstyle=\color{black}\bfseries
}

\lstdefinelanguage{mylanguage}
{
  basicstyle=\footnotesize\ttfamily,
  morecomment=[l][\color{black}]{*},
  morecomment=[l][\color{Mahogany}]{\#},
  morecomment=[l][\color{Mahogany}]{//},
  morecomment=[n][\bfseries]{/*}{*/},
  commentstyle=\color{black}\bfseries
}

  % morecomment=[n][\color{LimeGreen}]{/*}{*/},

% \usepackage{color}   %May be necessary if you want to color links
\usepackage{hyperref}  % Clickable table of content
\hypersetup{
    colorlinks=false, %set true for want colored links
    linktoc=all,      %set to all for linking both sections and subsections
    % linkcolor=blue,   %choose some color if you want links to stand out
}
\parindent0pt
\parskip 1.5ex plus 1ex minus .5ex

\definecolor{Blue}{rgb}{0,0,1}                                                     
\definecolor{Red}{rgb}{1,0,0}                                                      
\definecolor{Green}{rgb}{0,1,0}                                                    
\definecolor{Bronze}{rgb}{0.8,0.5,0.2}                                             
\definecolor{Violet}{rgb}{0.54,0.17,0.89}                                          
                                                                                   
\newcommand{\TODO}[1]{{\textcolor{Violet}{TODO: #1}}}                              
\newcommand{\YC}[1]{{\textcolor{Bronze}{#1}}}                                     
\newcommand{\SG}[1]{{\textcolor{Blue}{#1}}}

\DeclareMathOperator{\Tr}{Tr}
\newcommand{\Ell}{\mathcal{L}}
\newcommand{\R}{\mathds{R}}
\newcommand{\N}{\mathds{N}}
\newcommand{\C}{\mathds{C}}
\newcommand{\bfa}{\boldsymbol{\alpha}}
\newcommand{\bs}[1]{{\boldsymbol{#1}}}


\newcommand\blfootnote[1]{%
  \begingroup
  \renewcommand\thefootnote{}\footnote{#1}%
  \addtocounter{footnote}{-1}%
  \endgroup
}

\title{Quandary: Optimal Control for Open and Closed Quantum Systems}
\author{Stefanie G{\"u}nther\thanks{Center for Applied Scientific Computing, Lawrence Livermore National Laboratory, Livermore, CA, USA.} \and N. Anders Petersson$^*$} 
% \author{Stefanie G{\"u}nther \and N. Anders Petersson} 
\date{last updated \today}

\begin{document}
\maketitle


\section*{Installation}
 Read the \texttt{README.md}! In short:
 \begin{enumerate}
  \item Install Petsc (\url{https://petsc.org/}).
  \item Compile the quandary executable and install with
  \begin{itemize}
    \item[$>$] \texttt{mkdir build \&\& cd build}
    \item[$>$] \texttt{cmake ..}
    \item[$>$] \texttt{make}
    \item[$>$] \texttt{sudo cmake --install .}
  \end{itemize}
  \item To use the python interface, create a virtual environment and do:
  \begin{itemize}
    \item[$>$] \texttt{pip install -e .}
  \end{itemize}
 \end{enumerate}

 
 \subsection*{Quick start} 
 
The C++ Quandary executable takes a configuration input file. As a quick start, test it with 
\begin{itemize}
  \item[$>$]  \texttt{./quandary config\_template.cfg}  \hfill (serial execution)
  \item[$>$]  \texttt{mpirun -np 4 ./quandary config\_template.cfg}  \hfill (on 4 cores)
\end{itemize}
You can silence Quandary by adding the \texttt{--quiet} command line argument.

Results are written as column-based text files in the output directory. Gnuplot is an excellent plotting tool to visualize the written output files, see below. The \texttt{config\_template.cfg} is currently set to run a CNOT optimization test case. It lists all available options and configurations, and is filled with comments that should help users to set up new simulation and optimization runs, and match the input options to the equations found in this document. 

Test the python interface by running one of the examples in \texttt{examples/}, e.g. 
\begin{itemize}
  \item[$>$]  \texttt{python3 example\_swap02.py}
\end{itemize}
 The python interpreter will start background processes on the C++ executable using a config file written by the python interpreter, and gathers quandary's output results back into the python shell for plotting.

\tableofcontents

\section{Introduction}
Quandary numerically simulates and optimizes the time evolution of closed and open quantum systems. The
underlying dynamics are modelled by either Schroedinger's equation (for closed systems), or Lindblad's master equation (for open systems that interact with the environment). Quandary solves the respective ordinary differential equation (ODE) numerically by applying a time-stepping integration scheme, and applies a gradient-based optimization
scheme to determine optimal control pulses that drive the quantum system to a desired target.
The target can be a unitary, i.e. optimizing for pulses that
realize a logical quantum operation, or state preparation that aims to drive the quantum system from one (or multiple) initial state to a desired target state, such as for example the ground state of zero energy level, or for the creation of entangled state. 

Quandary is designed to solve optimal control problems in larger (potentially open) quantum systems, targeting modern high performance computing (HPC) platforms. Quandary utilizes distributed memory computations using the message passing paradigm that enables scalability to large number of compute cores. Implemented in C++, Quandary is portable and its object-oriented implementation allows developers to extend the predefined setup to suit their particular simulation and optimization requirements. For example, customized gates for Hamiltonian simulations can easily be added to supplement Quandary’s predefined gate set. 
The Python interface allows for greater flexibility where custom Hamiltonian models can be used.

This document outlines the mathematical background and underlying equations, and summarizes their
implementation and usage in Quandary. We also refer to our publications \cite{guenther2021quandary, guenther2021quantum}.


\section{Model equation}\label{sec:model}
Quandary models composite quantum systems consisting of $Q$ subsystems, with $n_k$ energy levels for the
$k$-th subsystem, $k=0,\dots,Q-1$. The Hilbert space dimension is hence the product of each subsystem dimensions: $N = \prod_{k=0}^{Q-1} n_k$. 

The default system Hamiltonian model for the composite system is
\begin{align}
  H_d &:= \sum_{k=0}^{Q-1} \left(\omega_k a_k^{\dagger}a_k- \frac{\xi_k}{2} a_k^{\dagger}a_k^{\dagger}a_k a_k  + \sum_{l> k} \left(  J_{kl} \left( a_k^\dagger a_l + a_k a_l^\dagger \right) -\xi_{kl} a_{k}^{\dagger}a_{k}   a_{l}^{\dagger} a_{l} \right)\right)
\end{align}
where $\omega_k\geq 0$ denotes the $0 \rightarrow 1$ transition frequency and $\xi_k\geq 0$ is the self-Kerr coefficient of subsystem $k$, and the cross resonance coefficients are $J_{kl}\geq 0$ (``dipole-dipole interaction'') and $\xi_{kl}\geq 0$ (``zz-coupling''). Here,
$a_k\in \C^{N\times N}$ denotes the lowering operator acting on subsystem $k$, which is defined as
\begin{align}
  \begin{array}{rl}
  a_0 &:= a^{(n_0)} \otimes I_{n_1} \otimes \dots \otimes
  I_{n_{Q-1}}\\
  a_1 &:= I_{n_0} \otimes a^{(n_1)} \otimes \dots \otimes
  I_{n_{Q-1}}\\
  \vdots \, & \\
  a_{Q-1} &:= I_{n_0} \otimes I_{n_1} \otimes \dots \otimes
  a^{(n_{Q-1})}\\
  \end{array}
  \quad \text{with}\quad
 a^{(n_k)} := \begin{pmatrix}
   0 & 1 &          &         &    \\
     & 0 & \sqrt{2} &         &     \\
     &   & \ddots   & \ddots  &    \\
     &   &          &         & \sqrt{n_k-1}  \\
     &   &          &         & 0   
 \end{pmatrix} \in \R^{n_k \times n_k}
\end{align}
where $I_{n_k} \in \R^{n_k \times n_k}$ is the identity matrix.



The action of external control fields on the quantum system is modelled through the control Hamiltonian 
\begin{align}
  H_c(t) &:= \sum_{k=0}^{Q-1} f^k(\vec{\alpha}^k,t) \left(a_k + a_k^\dagger \right)
\end{align}
where $f^k(\vec{\alpha}^k,t)$ are real-valued, time-dependent control functions that are parameterized by real-valued parameters $\vec{\alpha}^k\in \R^d$, which can be either specified, or optimized for. 

For a \textbf{closed quantum system} (no environmental interactions), the quantum state is described by a complex-valued vector $\psi\in\C^N$, with $\|\psi\| = 1$. For a given initial state $\psi(t=0)$, the evolution of the state vector is modelled through \textbf{Schroedinger's equation}
\begin{align} \label{eq:schroedinger}
  \dot \psi(t)  = -i H(t) \psi(t), \quad \text{with} \quad  H(t) := H_d + H_c(t).
\end{align}

\textbf{Open quantum systems} take interactions with the environment into account, allowing us to model decoherence and noise in the system. In that case, the state of the quantum system is described by its density matrix $\rho\in \C^{N\times N}$, and the time-evolution is modelled by \textbf{Lindblad's master equation}:
\begin{align}\label{mastereq}
  \dot \rho(t) = &-i(H(t)\rho(t) - \rho(t)H(t)) + \Ell(\rho(t)),
\end{align}
where again $H(t) = H_d + H_c(t)$, and where $\Ell(\rho(t))$ denotes the Lindbladian collapse operators to model system-environment interactions. The Lindbladian operator $\Ell(\rho(t))$ is assumed to be of the form 
\begin{align} \label{eq:collapseop}
  \Ell(\rho(t)) = \sum_{k=0}^{Q-1} \sum_{l=1}^2 \Ell_{lk} \rho(t)
  \Ell_{lk}^{\dagger} - \frac 1 2 \left( \Ell_{lk}^{\dagger}\Ell_{lk}
  \rho(t) + \rho(t)\Ell_{lk}^{\dagger} \Ell_{lk}\right)
\end{align}
where the collapse operators $\Ell_{lk}$ model decay and dephasing processes in the subsystem $k$ with 
\begin{itemize}
  \item  Decay  (``$T_1$''): $\Ell_{1k} = \frac{1}{\sqrt{T_1^k}} a_k$
  \item  Dephasing  (``$T_2$''): $\Ell_{2k} = \frac{1}{\sqrt{T_2^k}} a_k^{\dagger}a_k$ 
\end{itemize}
The constants $T_l^k>0$ correspond to the half-life of process $l$ on subsystem $k$. Typical $T_1$ decay time is between $10-100$ microseconds (us). $T_2$ dephasing time is typically about half of T1 decay time. 
% Decay processes typically behave like $\exp(-t/{T_1})$.

All the above constants and system parameters can be specified in the first part of the configuration file that Quandary's executable takes as an input, compare \texttt{config\_template.cfg}. 
Note that the main choice here is which equation should be solved for and which representation of the quantum state will be used (either Schroedinger with a state vector $\psi \in \C^N$, or Lindblad's equation for a density matrix $\rho \in \C^{N\times N}$). In the configuration file, this choice is determined through the option \texttt{collapse\_type}, where \texttt{none} will result in Schroedinger's equation and any other choice will result in Lindblad's equation being solved for. Further note, that choosing \texttt{collapse\_type} $\neq$ \texttt{none}, together with a collapse time $T_{l}^k = 0.0$ will omit the evaluation of the corresponding term in the Lindblad operator \eqref{eq:collapseop} (but will still solve Lindblad's equation for the density matrix).

\textit{Note:} In the remainder of this document, the quantum state will mostly be denoted by $\rho$, independent of which equation is solved for. Depending on the context, $\rho$ can then either denotes the density matrix $\rho\in \C^{N\times N}$, or the state vector $\psi\in \C^N$. A clear distinction between the two will only be made explicit if necessary.

\subsection{Rotational frame approximation}
Quandary uses the rotating wave approximation in order to slow down the time scales in the solution of Schroedinger's or Lindblad's master equations. To that end, the user can specify the rotation frequencies $\omega_k^r$ for each oscillator. Under the rotating frame wave approximation, the Hamiltonians are transformed to 
\begin{align} 
  \tilde H_d(t) &:= \sum_{k=0}^{Q-1} \left(\omega_k - \omega_k^{r}\right)a_k^{\dagger}a_k- \frac{\xi_k}{2}
  a_k^{\dagger}a_k^{\dagger}a_k a_k  
   - \sum_{l> k} \xi_{kl} a_{k}^{\dagger}a_{k}   a_{l}^{\dagger} a_{l} \notag \\
   & + \sum_{k=0}^{Q-1}\sum_{l>k} J_{kl} \left(\cos(\eta_{kl}t) \left(a_k^\dagger a_l + a_k a_l^\dagger\right) + i\sin(\eta_{kl}t)\left(a_k^\dagger a_l - a_k a_l^\dagger\right) \right) \label{eq:Hd_rotating} \\
   %
   \tilde H_c(t) &:= \sum_{k=0}^{Q-1} \left( p^k(\vec{\alpha}^k,t) (a_k +
   a_k^{\dagger}) + i q^k(\vec{\alpha}^k,t)(a_k - a_k^{\dagger})
   \right)  \label{eq:Hc_rotating}
\end{align} 
where $\eta_{kl} := \omega_k^{r} - \omega_l^{r}$ are the differences in rotational frequencies between subsystems. 

Note that the eigenvalues of the rotating frame Hamiltonian become significantly smaller in magnitude by choosing $\omega_k^r \approx \omega_k$ (so that the first term with $a_k^\dagger a_k$ drops out). This slows down the time variation of the state evolution, hence bigger time-step sizes can be chosen when solving the master equation numerically. We remark that the rotating wave approximation ignores terms in the control Hamiltonian that oscillate with frequencies $\pm 2\omega_k^r$. Below, we drop the tildes on $\tilde H_d$ and $\tilde H_c$ and use the rotating frame definition of the Hamiltonians to model the system evolution in time. 

Using the rotating wave approximation, the real-valued laboratory frame control functions are written as 
\begin{align}
  f^k(\vec{\alpha}^k,t) = 2\mbox{Re}\left(d^k(\vec{\alpha}^k,t)e^{i\omega_k^r t}\right), \quad d^k(\vec{\alpha}^k,t) = p^k(\vec{\alpha}^k,t) + i q^k(\vec{\alpha}^k,t)
\end{align}
where the rotational frequencies $\omega_k^r$ act as carrier waves to the rotating-frame control functions $d^k(\vec{\alpha}^k, t)$. 


\subsection{Control pulses} \label{subsec:controlpulses}
The time-dependent rotating-frame control functions $d^k(\vec{\alpha}^k,t)$ are parameterized using $N_s^k$ basis functions $B_s(t)$ acting as envelope for $N_f^k$ carrier waves:
\begin{align}\label{eq:spline-ctrl}
  d^k(\vec{\alpha}^k,t) = \sum_{f=1}^{N_f^k} \sum_{s=1}^{N_s^k} \alpha_{s,f}^k B_s(t) e^{i\Omega_k^ft}, \quad \alpha_{s,f}^k = \alpha_{s,f}^{k(1)} + i \alpha_{s,f}^{k(2)} \in \C
\end{align}
By default, the basis functions are piecewise quadratic (2nd order) B-spline polynomials, centered on an equally spaced grid in time. To instead use a piecewise constant (0th order) B-spline basis, see Section~\ref{subsec:bspline-0}.
The amplitudes $\alpha_{s,f}^{k(1)}, \alpha_{s,f}^{k(2)} \in \R$ are the control
parameters (\textit{design} variables) that Quandary can optimize in order to realize a
desired system behavior, giving a total number of $2\sum_k N_s^k N_f^k$ real-valued optimization variables. (Note that the number of carrier wave frequencies $N_f^k$ as well as the number of spline basis functions $N_s^k$ can be different for each subsystem $k$.) $\Omega_k^f \in \R$ denote the carrier wave frequencies in the rotating frame which can be chosen to trigger certain system frequencies. The corresponding Lab-frame carrier frequencies become $\omega_k^r + \Omega_k^f$. Those frequencies can be chosen to match the transition frequencies in the lab-frame system Hamiltonian. For example, when $\xi_{kl} << \xi_k$, the transition frequencies satisfy $\omega_k - n\xi_k$. Thus by choosing $\Omega_k^f = \omega_k-\omega_k^r - n \xi_k$, one triggers transition between energy levels $n$ and $n+1$ in subsystem $k$. Choosing effective carrier wave frequencies is quite important for optimization performance. We recommend to have a look at \cite{petersson2021optimal} for details on how to choose them.

Using trigonometric identities, the real and imaginary part of the rotating-frame control $d^k(\vec{\alpha}^k,t) = p^k(\vec{\alpha}^k,t) + iq^k(\vec{\alpha}^k,t)$ are given by
\begin{align}
  p^k(\vec{\alpha}^k,t) &= \sum_{f=1}^{N_f^k} \sum_{s=1}^{N_s^k} B_s(t)
  \left(\alpha^{k
  (1)}_{s,f} \cos(\Omega_f^k t) - \alpha^{k (2)}_{s,f} \sin(\Omega_f^k t)
  \right) \\
  q^k(\vec{\alpha}^k,t) &= \sum_{f=1}^{N_f^k} \sum_{s=1}^{N_s^k} B_s(t)\left( \alpha^{k
  (1)}_{s,f} \sin(\Omega_f^k t) + \alpha^{k (2)}_{s,f} \cos(\Omega_f^k t)
  \right)
\end{align}
Those relate to the Lab-frame control $f^k(\vec{\alpha}^k,t)$ through
\begin{align}
  f^k(t) &=  2\sum_{f=1}^{N_f^k} \sum_{s=1}^{N_s^k} B_s(t) \left(\alpha_{s,f}^{k(1)} \cos((\omega_k^{r} + \Omega_f^k) t) - \alpha_{s,f}^{k(2)}\sin((\omega_k^{r} + \Omega_f^k) t) \right) \\
         &= 2 p^k(\vec{\alpha}^k, t) \cos(\omega_k^{r} t) - 2 q^k(\vec{\alpha}^k,
         t)\sin(\omega_k^{r} t) \\
         &= 2\mbox{Re}\left( d^k(\vec{\alpha}^k,t)e^{i\omega_k^r t} \right)
\end{align}

\subsubsection{Storage of the control parameters}

The control parameters $\bs{\alpha}$ are stored in the Quandary code in the following order: List oscillators first $(\vec{\alpha}^0, \dots, \vec{\alpha}^{Q-1})$, then for each $\vec{\alpha}^k \in
\R^{2 N_s^k N_f^k}$, iterate over all carrierwaves $\vec{\alpha}^k =
(\alpha^k_1,\dots, \alpha^k_{N_f})$ with $\alpha^k_f \in \R^{2 N_s^k}$, then each
$\alpha^k_f$ iterates over spline basis functions listing first all real then all imaginary 
components: $\alpha^k_f = \alpha^{k(1)}_{1,f}, \dots, \alpha^{k(1)}_{N_s^k,f}, \alpha^{k(2)}_{1,f}, \dots, \alpha^{k(2)}_{N_s^k,f}$. Hence there are a total of $2\sum_k N_s^k N_f^k$ real-valued optimization parameters, which are stored in the following order:
  \begin{align}
    \boldsymbol{\alpha} &:= \left( \vec{\alpha}^0, \dots, \vec{\alpha}^{Q-1} \right), \in
    \mathds{R}^{2\sum_k N_s^k N_f^k} \quad \text{where}\\
    \vec{\alpha}^k = &\left( \alpha_{1,1}^{k(1)},\dots, \alpha_{N_s^k,1}^{k(1)}, \dots, \alpha_{1,N_f^k}^{k(1)}, \dots, \alpha_{N_s^k,N_f^k}^{k(1)} \right.\\
                   &  \left. \alpha_{1,1}^{k(2)},\dots, \alpha_{N_s^k,1}^{k(2)}, \dots, \alpha_{1,N_f^k}^{k(2)}, \dots, \alpha_{N_s^k,N_f^k}^{k(2)} \right)
  \end{align}
  iterating over $Q$ subsystems first, then $N_f^k$ carrier wave frequencies, then $N_s^k$ splines, listing first all real parts then all imaginary parts. To access an element $\alpha_{s,f}^{k(i)}$, $i=0,1$, from storage $\bfa$:
  \begin{align}
    \alpha_{s,f}^{k(i)} = \bfa \left[ \left(\sum_{j=0}^{k-1} 2 N_s^j N_f^j\right) + f*2 N_s^k + s + i*N_s^k N_f^k \right],
  \end{align}
  \textit{Note: this ordering of the controls is compatible with the order of control parameters in the Juqbox.jl software \cite{petersson2021optimal}.}

  When executing Quandary, the control parameter $\boldsymbol{\alpha}$ can be either specified (e.g. a constant pulse, a pi-pulse, or 
  pulses whose parameters are read from a given file), or can be optimized for (Section \ref{sec:optim}). 
  
  In order to guarantee that the optimizer yields control pulses that are
  bounded with $|p^k(t)| \leq c^k_{max}$, $|q^k(t)| \leq c^k_{max}$ for given bounds $c^k_{max}$ for each  
  subsystem $k=0,\dots, Q-1$, box constraints are implemented as:
   \begin{align}
     | \alpha_{s,f}^{k(1)}| \leq \frac{c^k_{max}}{N_f^k} \quad \text{and} \quad |
     \alpha_{s,f}^{k(2)} | \leq \frac{c^k_{max}}{N_f^k}.
   \end{align}

  \subsubsection{Alternative control parameterization based on B-spline amplitudes and time-constant phases}
  As an alternative to the above parameterization, we can parameterize only the \textit{amplitudes} of the control pulse with B-splines, and add a time-constant phase per carrierwave:
  \begin{align}
    d(t) = \sum_f e^{i\Omega_f t} a_f(t)e^{ib_f} \quad \text{where} \quad a_f(t) = \sum_s \alpha_{f,s} B_s(t) \\
    \Rightarrow d(t)= \sum_f\sum_s \alpha_{f,s}B_s(t)e^{i\Omega_ft + b_f}
  \end{align}
  where the control parameters are $b_f\in [-\pi, \pi]$ (phases for each carrier wave) and the amplitudes $\alpha_{f,s}\in \R$ for $s=1,\dots, N_s$, $f=1,\dots, N_f$. Hence for $Q$ oscillators, we have a total of $\sum_q (N_s^q + 1) N_f^q$ control parameters.

  The rotating frame pulses are then given by 
  \begin{align}
    p(t) = \sum_f \sum_s \alpha_{f,s} \cos(\Omega_f t + b_f) B_s(t) \\
    q(t) = \sum_f \sum_s \alpha_{f,s} \sin(\Omega_f t + b_f) B_s(t)
  \end{align}

\subsubsection{Zeroth order B-spline basis functions (piecewise constant controls)}\label{subsec:bspline-0}
A piecewise continuous envelope function can be generated by using zeroth order B-spline basis functions. When the carrier wave frequency is set to zero, this results in a control function that is piecewise constant in the rotating frame. For example, to use the zeroth order basis functions for controlling sub-system number 0 with 50 constant control segments, use the configuration option:
\begin{verbatim} 
  control_segments0 = spline0, 50
\end{verbatim}
When optimizing with zeroth order B-spline control functions, strong variations between consecutive control amplitudes can be avoided by enabling the total variation penalty term through the command
\begin{verbatim} 
  optim_penalty_variation= 1.0
\end{verbatim}
Compare Section \ref{sec:penalty}.


\subsection{Interfacing to Python environment}
You can use the Python interface for Quandary to simulate and optimize from within a python environment (version $\geq$ 3). It eases the use of Quandary, and adds some additional functionality, such as automatic computation of the required number of time-steps, automatic choice of the carrier frequencies, and it allows for custom Hamiltonian models to be used (system and control Hamiltonian operators $H_d$ and $H_c$). A good place to start is to have a look into the example \texttt{example\_swap02.py}. This test case optimizes for a 3-level SWAP02 gate that swaps the state of the zero and the second energy level of a 3-level qudit. 

All interface functions are defined in \texttt{quandary.py}. Most importantly, it defines the \texttt{Quandary} dataclass that gathers all configuration options and sets defaults. Default values are overwritten by user input either through the constructor call through \texttt{Quandary(<membervar>=<value>)} directly, or by accessing the member variables after construction and calling \texttt{update()} afterwards (compare \texttt{example\_swap02.py}). 

After setting up the configuration, you can evoke simulations or optimizations with \texttt{quandary.\-simulate/\-optimize()}. Check out \texttt{help(Quandary)} to see all available user functions. Under the hood, those function writes the required Quandary configuration files (\texttt{config.cfg}, etc.) to a data directory, then evokes (multiple) subprocesses to execute parallel C++ Quandary on that configuration file through the shell, and then loads the results from Quandary's output files back into the python interpreter. Plotting scripts are also provided, see the example scripts.

In addition to the standard Hamiltonian models as described in Section \ref{sec:model}, the python interface allows for user-defined Hamiltonian operators $H_d$ and $H_c$. Those are provided to Quandary through optional arguments to the python configuration \texttt{Quandary}, in which case Quandary replaces the Hamiltonian operators in \eqref{eq:Hd_rotating} (system Hamiltonian) and \eqref{eq:Hc_rotating} (control Hamiltonian operators $a\pm a'$) by the provided matrices.
\begin{itemize}
  \item The system Hamiltonian $H_d$ is a time-independent complex hermitian matrix. The units of the system Hamiltonian should be angular frequency (multiply $2\pi$). 
  \item For each oscillator, one complex-valued control operator can be specified. Those should be provided in terms of their real and imaginary parts separately, e.g. the standard model control operators would be specified as $H_{c,k}^{re} = a_k+a_k^\dagger$ and $H_{c,k}^{im}=a_k-a_k^\dagger$, for each oscillator $k$. The real parts will be multiplied by the control pulses $p_k(t)$, while the imaginary parts will be multiplied by $iq_k(t)$ for each oscillator $k$, similar to the model in \eqref{eq:Hc_rotating}. Control Hamiltonian operators should be 'unit-free', since those units come in through the multiplied control pulses $p$ and $q$.
  \item To enable the use of the custom Hamiltonians, pass the configuration option \texttt{standardmodel=False}, in addition to \texttt{Hsys=<yourSystemHamiltonian>} and \texttt{Hc\_real=[<HcReal oscillator1, HcReal oscillator2, ...]},  \texttt{Hc\_imag=[<HcImag oscillator1, HcImag oscillator2, ...]}. 
  \item Note: The control Hamiltonian operators are optional, but the system Hamiltonian is always required if \texttt{standardmodel=False}. 
  \item Note: The matrix-free solver can not be used when custom Hamiltonians are provided. The code will therefore be slower. 
\end{itemize}

The python interface is set up such that it automatically computes the time-step size for discretizing the time domain, as well as the carrier wave frequencies that trigger system resonances. Note that the carrier wave frequency analysis are tailored for the standard Hamiltonian model, and those frequencies might need to be adapted when custom Hamiltonian operators are used (read the screen output). You can always check the written configuration file \texttt{config.cfg}, and the log to see what frequencies are being used, and potentially modify them. 

To switch between the Schroedinger solver and the Lindblad solver, the optional $T_1$ decay and $T_2$ dephasing times can be passed to the python QuandaryConfig. For the Lindblad solver, the same collapse terms as defined in \eqref{eq:collapseop} will be added to the dynamical equation. 

\section{The Optimal Control Problem} \label{sec:optim}
In the most general form, Quandary can solve the following optimization problem
\begin{align}\label{eq:minproblem}
  % \min_{\boldsymbol{\alpha}} \frac{1}{n_{init}} \sum_{i=1}^{n_{init}} \beta_i J(\rho^{target}_i, \rho_i(T)) + \gamma_1 \int_0^T P\left(\rho_i(t)\right) \, \mathrm{d} t + \gamma_2 \| \bfa \|^2_2
  \min_{\boldsymbol{\alpha}} J(\{\rho^{target}_i, \rho_i(T) \}) \quad + \mbox{Regularization} + \mbox{Penalty}
\end{align}
where $\rho_i(T)$ denotes one or multiple quantum states evaluated at a final time $T>0$, which solve either Lindblad's master equation \eqref{mastereq} or Schroedinger's equation \eqref{eq:schroedinger} in the rotating frame for initial conditions $\rho_i(0)$, as specified in Section \ref{subsec:initcond}, $i=1,\dots, n_{init}$. The first term in \eqref{eq:minproblem} minimizes an objective function $J$ (see Section \ref{sec:objectivefunctionals}) that quantifies the discrepancy between the realized states $\rho_i(T)$ at final time $T$ driven by the current control $\boldsymbol{\alpha}$ and the desired target $\rho^{target}_i$, see Section \ref{sec:targets}. 
The remaining terms are regularization and penalty terms that can be added to stabilize convergence, or prevent leakage, compare Section \ref{sec:penalty}

\subsection{Fidelity}\label{sec:fidelity}

As a measure of optimization success, Quandary reports on the fidelity computed from 
\begin{align}\label{eq:fidelity}
  F = \begin{cases}
    \frac{1}{n_{init}} \sum_{i=1}^{n_{init}} \mbox{Tr}\left(\left(\rho^{target}_i\right)^\dagger\rho_i(T) \right) & \text{if Lindblad} \\
    \left|\frac{1}{n_{init}} \sum_{i=1}^{n_{init}} (\psi^{target}_i)^\dagger \psi_i(T) \right|^2 & \text{if Schroedinger}
  \end{cases}
\end{align}
The fidelity is an average of Hilbert-Schmidt overlaps of the target states and the evolved states: for the density matrix, the Hilbert-Schmidt overlap is $\langle \rho^{target}, \rho(t)\rangle = \mbox{Tr}\left(\left(\rho^{target}\right)^\dagger\rho(T)\right)$, which is \textit{real} if both states are density matrices (which is always the case in Quandary, see definition of basis matrices). For the state vector (and the Schroedinger solver), the Hilbert-Schmidt overlap is $\langle \psi^{target}, \psi(T)\rangle = (\psi^{target})^{\dagger}\psi $, which is complex. Note that in the fidelity above (and also in the corresponding objective function $J_{trace}$, the absolute value is taken \textit{outside} of the sum, hence relative phases are taken into account. 

Further note that this fidelity is averaged over the chosen initial conditions, so the user should be careful how to interpret this number. E.g. if one optimizes for a logical gate while choosing the three initial condition as in Section \ref{subsec:threeinitcond}, the fidelity that is reported during optimization will be averaged over those three initial states, which is not sufficient to estimate the actual average fidelity over the entire space of potential initial states. It is advised to recompute the average fidelity \textbf{after} optimization has finished by propagating all basis states $B_{kj}$ to final time $T$ using the optimized control parameter, or by propagating only $N+1$ initial states to get an estimate thereof.


\subsection{Objective function}\label{sec:objectivefunctionals}
The following objective functions can be used for optimization in Quandary (config option \texttt{optim\_objective}):
\begin{align}
 J_{Frobenius} &= \sum_{i=1}^{n_{init}} \frac{\beta_i}{2} \left\| \rho^{target}_i - \rho_i(T)\right\|^2_F \\ 
 J_{trace} &= 
\begin{cases} 
 1 - \sum_{i=1}^{n_{init}} \frac{\beta_i}{w_i} \mbox{Tr}\left((\rho^{target}_i)^\dagger\rho_i(T)\right) & \text{if Lindblad}\\
 1 - \left|\sum_{i=1}^{n_{init}} \beta_i (\psi^{target}_i)^\dagger\psi_i(T)\right|^2 & \text{if Schroedinger}
\end{cases}\\
 J_{measure} &= \sum_{i=1}^{n_{init}} \beta_i \mbox{Tr} \left( N_m \rho(T) \right) \label{eq:Jmeasure} 
\end{align}
Here, $\beta_i$ denote weights with $\sum_{i=1}^{n_{init}} \beta _i = 1$ that can be used to scale the contribution of each initial/target state $i$ (default $\beta_i = 1/n_{init}$). 
$J_{Frobenius}$ measures (weighted average of) the Frobenius norm between target and final states. $J_{Trace}$ measures the (weighted) infidelity in terms of the Hilbert-Schmidt overlap, compare the definition of fidelity in eq. \eqref{eq:fidelity}. Here, $w_i = \mbox{Tr}\left(\rho_i(0)^2\right)$ is the purity of the initial state. Both those measures are common for optimization towards a unitary gate transformation, for example. $J_{measure}$ is (only) useful when considering pure-state optimization, see Section \ref{sec:statepreparation}. Here, $m\in\N$ is a given integer, and $N_m$ is a diagonal matrix with diagonal elements being $|k-m|, k=0,\dots N-1$ 

The distinction for the Lindblad vs. Schroedinger solver is made explicit for $J_{trace}$ above. The other two measures apply naturally to either the density matrix version solving Lindblad's equation, or the state vector version solving Schroedinger's equation with $ \|\rho^{target} - \rho(T)\| = \|\psi^{target} - \psi(T)\|$ and $\mbox{Tr}\left(N_m\rho(T)\right) = \psi(T)^\dagger N_m \psi(T)$. 



\subsection{Optimization targets} \label{sec:targets}
Here we describe the target states $\rho^{target}$ that are realized in Quandary (C++ config option \texttt{optim\_target}). Two cases are considered: State preparation, where the target state is the same for all initial conditions, and gate optimization, where the target state is a unitary transformation of the initial condition. 


\subsubsection{Pure target states}\label{sec:statepreparation}
State preparation aims to drive the system from either one specific or from any arbitrary initial state to a common desired (fixed) target state. Quandary can optimize towards \textit{pure} target states of the form 
\begin{align}
 \psi^{target} = \boldsymbol{e}_m \quad \text{or} \quad \rho^{target} = \boldsymbol{e}_m \boldsymbol{e}_m^\dagger, \quad \text{for}\quad m\in \N_0\quad \text{with}\quad 0\leq m < N
\end{align}
where $\boldsymbol{e}_m$ denotes the $m$-th unit vector in $\R^N$. 
\footnote{We note that considering pure states of that specific form ($\boldsymbol{e}_m$ or $\boldsymbol{e}_m\boldsymbol{e}_m^\dagger$) is not a restriction, because any other pure target state can be transformed to this representation using a unitary change of coordinates (compare the Appendix in \cite{guenther2021quantum} for a more detailed description).}
The integer $m$ refers to the $|m\rangle$-th state of the entire system under consideration with dimension $N$, which can be a composite of $Q$ subsystems. 
In the configuration file however, the pure target state is specified by defining the desired pure target for \textit{each} of the subsystems individually. For a composite system of $Q$ subsystems with $n_k$ levels each, a composite target pure state is specified by a list of integers $m_k$ with $0\leq m_k < n_k$ representing the pure target state in each subsystem $k$. The composite pure target state is then
\begin{align}
  \psi^{target} = |m_0 m_1 m_2 \dots m_{Q-1} \rangle \quad \text{aka} \quad  \psi^{target} = \boldsymbol{e}_{m_0}\otimes  \boldsymbol{e}_{m_1}\otimes  \dots \otimes\boldsymbol{e}_{m_{Q-1}} 
\end{align}
for unit vectors $\boldsymbol{e}_{m_k} \in \R^{n_k}$, and $\rho^{target} = \psi^{target} (\psi^{target})^\dagger$ for the density matrix. The composite-system index $m$ is computed inside Quandary, from 
\begin{align}
  m = m_0 \frac{N}{n_0} + m_1 \frac{N}{n_0n_1} + m_2 \frac{N}{n_0n_1n_2} + \dots + m_{Q-1}
\end{align}

Depending on the choice for the initial conditions, optimization towards a pure target state can be used to realize either a simple state-to-state transfer (choosing one specific initial condition, $n_{init}=1$), or to realize the more complex task of state preparation that drives \textit{any} initial state to a common pure target state. 
For $m=0$, the target state represents the ground state of the system under consideration, which has important applications for quantum reset as well as quantum error correction. Driving \textit{any} initial state to a common target will require to couple to a dissipative bath, which should be accounted for in the model setup. In the latter case, typically a full basis of initial conditions needs to be considered during the optimization ($n_{init}=N^2$ for density matrices). However, it is shown in \cite{guenther2021quantum}, that if one chooses the objective function $J_{measure}$ with corresponding measurement operator $N_m$ (see eq. \eqref{eq:Jmeasure}), one can reduce the number of initial conditions to only \textit{one} being an ensemble of all basis states, and hence $n_{init}=1$ independent of the system dimension $N$. Compare \cite{guenther2021quantum} for details, and Section \ref{sec:ensemblestate}. 

% For example, assume one considers a composite system consisting of a qubit modelled with $n_0=2$ levels coupled to a readout cavity modelled with $n_1=10$ levels, and one wants to drive the qubit to the $|1\rangle$ state and the cavity to the ground state. The target pure state input for Quandary is hence $m_0=1, m_1=0$ (i.e. the $|10\rangle$ state in the composite system), which corresponds $m=1\cdot10+0 = 10$ in the composite system of dimension $N=2\cdot 10 = 20$. 


\subsubsection{Arbitrary target state}
A specific (non-pure) target state $\rho^{target}$ can also be used as a target. For the C++ code, such a target state is read from file. File format: The vectorized density matrix (column-wise vectorization) in the Lindblad case, or the state vector in the Schroedinger case, one real-valued number per row, first list all real parts, then list all imaginary parts (hence either $2N^2$ lines with one real number each, or $2N$ lines with one real number each). The configuration option should be pointing to the file location. For the Python interface, the target state can be passed as a numpy array. 
 
\subsubsection{Gate optimization}

Quandary can be used to realize logical gate operations. In that case, the target state is not fixed across the initial states, but instead is a unitary transformation of each initial condition. Let $V\in \C^{N\times N}$ be a unitary matrix presenting a logical operation, the goal is to drive any initial state $\rho(0)$ to the unitary transformation $\rho^{target} = V\rho(0)V^{\dagger}$, or, in the Schroedinger case, drive any initial state $\psi(0)$ to the unitary transformation $\psi(T) =  V\psi(0)$.
In the C++ code, some default target gates that are currently available:
\begin{align}
  V_{X} := \begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix} \quad
  V_{Y} := \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix} \quad
  V_{Z} := \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \quad 
  V_{Hadamard} := \frac{1}{\sqrt{2}} 
           \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \\
  V_{CNOT} := \begin{bmatrix} 1  & 0 & 0 & 0 \\ 
                               0  & 1 & 0 & 0 \\ 
                               0  & 0 & 0 & 1 \\ 
                               0  & 0 & 1 & 0 \\ 
                \end{bmatrix} \quad 
  V_{SWAP} := \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 1 \\
  \end{bmatrix} \quad
  V_{QFT} = ...
\end{align}
as well as a multi-qubit generalization of the SWAP and the CNOT gate for general $Q$ subsystems: The SWAP-0Q gate swaps the state of the first and the last qubit, while leaving all other oscillators in their respective initial state, and the CQNOT gate performs a NOT operation on the last qubit if all other qubits are in the one-state.
New, user-defined gates can be added to the C++ code by augmenting the \texttt{Gate} class in the corresponding \texttt{.cpp} and \texttt{.hpp} files.
The target gate matrix can also be read from file. The file is a simple text file that contains the vectorized target unitary matrix (column-wise vectorization), first all real parts, then all imaginary parts (giving a total of $2N^2$ real-valued numbers). It should be specified in the essential dimensions. 
For the python interface, the target unitary matrix is passed to Quandary through a numpy array.

For gate optimization, the first two objective function $J_{Frobenius}$ and $J_{trace}$ are appropriate. Since \textit{any} initial quantum state should be transformed by the control pulses, typically a basis for the initial states should be considered ($n_{init} = N$ for Schroedinger solver, and $n_{init}=N^2$ for Lindblad solver). In the Lindblad solver case, it has however been shown in \cite{goerz2014optimal} that it is enough to optimize with only three specific initial states ($n_{init} = 3$), independent of the Hilbert space dimension $N$. Those three states are set up in such a way that they can distinguish between any two unitary matrices in that Hilbert space. The three initial states are readily available in Quandary, see Section \ref{subsec:initcond}. Note that when optimizing with only those three initial states, it turns out that the choice of the weights $\beta_i$ that weight the contribution from each initial state in the overall objective function strongly influences the optimization convergence. For faster convergence, it is often beneficial to emphasize on the first of the three initial conditions ($\rho_1(0)$ in Section \ref{subsec:threeinitcond}), hence choosing $\beta_1$ (much) bigger than $\beta_2$ and $\beta_3$ (e.g. $\beta = \{20,1,1\}$ often works better than $\beta = \{1,1,1\}$, try it yourself). We refer to \cite{goerz2014optimal} for details. Note that the weights will be scaled internally such that $\sum_i \beta_i = 1$.

Target gates will by default be rotated into the computational frame (see Section \ref{sec:model}). Alternatively, the user can specify the rotation of the target gate through the configuration option \texttt{gate\_rot\_freq} (list of floats). 

\subsubsection{Essential and non-essential levels} \label{sec:essentiallevels}
It is often useful, to model the quantum system with more energy levels than the number of levels that the target gate is defined on. For example when optimizing for a SWAP gate on two qubits, with $V_{SWAP}\in\C^{4\times 4}$, one might want to model each qubit with more than two energy levels in order to (a) model the (infinite dimensional) system with more accuracy by including more levels and (b) allow the system to transition through higher energy levels in order to achieve the target at final time $T$. In that case, the \textit{essential} levels denote the levels that the target gate is defined on.
To this end, Quandary provides the option to specify the number of essential energy levels $n_k^e$ in addition to the number of energy levels $n_k$, where $n_k^e \leq n_k$ for each subsystem $k$. The quantum dynamics are then modelled with (more) energy levels with $N=\prod_k n_k$ and $\rho(t)\in \C^{N\times N}$ (or $\psi\in\C^N$), while the gate is defined in the essential level dimensions only: $V\in \C^{N_e \times N_e}, N_e=\prod_k n_k^e$. In the example above, $n^e_0=n^e_1=2$ and hence $V_{SWAP}\in \C^{4\times 4}$, but one can choose the number of energy levels $n_0$ and $n_1$ to be bigger than $2$ to when modelling the system dynamics. 
 
To compute the objective function at final time T, the essential-dimensional gate is projected upwards to the full dimensions $\tilde V \in \C^{N\times N}$ by inserting identity blocks for rows/columns that correspond to a non-essential level of either of the subsystems. Hence, a realization of the gate $\tilde V$ will not alter the occupation of higher (non-essential) energy level compared to their initial occupation at $t=0$. 

% When only the three states as initial conditions are considered (see below), those three initial states will be spanned in the full dimensional system. On the other hand, when the basis of initial states is considered, or their diagonals only, those initial states will be spanned only in the essential level dimensions and zero rows and columns will be inserted for all non-essential levels. Hence, the gate will be realized for any initial state that is spanned in the essential level dimensions, and occupations of non-essential levels at $T=0$ are avoided. 
 


\subsection{Initial conditions}\label{subsec:initcond}
The initial states $\rho_i(0)$ which are accounted for during optimization in eq. \eqref{eq:minproblem} can be specified with the configuration option \texttt{initialcondition}. Below are the available choices. 

\subsubsection{Pure-state initialization}
One can choose to simulate and optimize for only one specific pure initial state (then $n_{init} = 1$). The initial density matrix is then composed of Kronecker products of pure states for each of the subsystems. E.g. for a bipartite system with $n_1
\otimes n_2$ levels, one can propagate any initial pure state 
\begin{align}
  \psi(0) &= |m_1\rangle \otimes |m_2\rangle \quad \text{for} \, m_1 \in \{0,\dots, n_1-1\}, m_2\in \{0,\dots, n_2-1\}\\
   \text{and} \quad \rho(0) &= \psi(0)\psi(0)^\dagger 
\end{align}
Note that, again, in this notation $|m_1\rangle = \boldsymbol{e}_{m_1} \in \R^{n_1}$. The configuration input takes a list of the integers $m_k$ for each subsystem. 

\subsubsection{Basis states}


To span any possible initial state, an entire basis of states can be used as initial conditions. For open systems using the density matrix representation (Lindblad solver), the $n_{init}=N^2$ basis states as defined in \cite{guenther2021quantum} are implemented:
\begin{align}\label{eq:basismats}
B^{kj} := \frac 12 \left(\bs{e}_k\bs{e}_k^\dagger + \bs{e}_j\bs{e}_j^\dagger\right) +  \begin{cases} 
          0 & \text{if} \, k=j \\ 
        \frac 12 \left( \bs{e}_k\bs{e}_j^\dagger  + \bs{e}_j\bs{e}_k^\dagger \right) & \text{if} \, k<j \\
        \frac i2 \left( \bs{e}_j\bs{e}_k^\dagger  - \bs{e}_k\bs{e}_j^\dagger \right) & \text{if} \, k>j
      \end{cases} 
\end{align}
for all $k,j\in\{0,\dots, N-1\}$.
These density matrices represent $N^2$ pure, linear independent states that span the space of all density matrices in this Hilbert space. For closed systems using the state vector representation (Schroedinger's solver), the basis states are the unit vector in $\C^{N}$, hence $n_{init} = N$ initial states $\boldsymbol{e}_i \in \R^N, i=0,\dots N-1$. 

When composite systems of multiple subsystems are considered, the user can provide a consecutive list of integer ID's to determine in which of the subsystems the basis states should be spanned. Other subsystems will then be initialized in the ground state.

\textit{Note:} The basis states are spanned in the \textit{essential dimensions} of the system, if applicable. 

In order to uniquely identify the different initial conditions in the Quandary code and in the output files, a
unique index $i \in \{0,\dots, N^2-1\}$ is assigned to each basis state with 
\begin{align*}
  B^i := B^{k(i), j(i)}, \quad \text{with} \quad k(i) := i \,\mbox{mod}\, N,
  \quad \text{and} \quad j(i) := \left\lfloor \frac{i}{N} \right\rfloor
\end{align*}
(column-wise vectorization of a matrix of matrices $\left\{B^{kj}\right\}_{kj}$). 



\subsubsection{Only the diagonal density basis matrices}

For density matrices (Lindblad solver), one can choose to propagate only those basis states that correspond to pure states of the form $\boldsymbol{e}_k\boldsymbol{e}_k^\dagger$, i.e. propagating only the $B^{kk}$ in \eqref{eq:basismats} for $k=0,\dots, N-1$, and then $n_{init}=N$. For the Schroedinger solver, this is equivalent to all basis states. 

Again, when composite systems of multiple subsystems are considered, the user can provide a consecutive list of integer ID's to determine in which of the subsystems the diagonal states should be spanned. Other subsystems will then be initialized in the ground state.

\textit{Note:} the diagonal states are spanned in the \textit{essential dimensions} of the system, if applicable. 



\subsubsection{Ensemble state for pure-state optimization}\label{sec:ensemblestate}
\textit{Only valid for the density matrix version, solving Lindblad's master equation.}

For pure-state optimization using the objective function $J_{measure}$ \eqref{eq:Jmeasure}, one can use the ensemble state 
\begin{align}\label{eq:ensemblestate}
  \rho_s(0) = \frac{1}{N^2}\sum_{i,j=0}^{N-1} B^{kj}
\end{align}
as the only initial condition for optimization or simulation ($\Rightarrow n_{init}=1$). Since Lindblad's master equation is linear in the initial condition, and $J_{measure}$ is linear in the final state, propagating this single initial state yields the same target value as if one propagates all basis states spanning that space and averages their measure at final time $T$ (compare \cite{guenther2021quantum}). To specify the ensemble state in Quandary for composite quantum systems with multiple subsystems, on can provide a list of integer ID's that determine in which of the subsystems the ensemble state should be spanned. Other subsystems will be initialized in the ground state. 

To be precise: the user specifies a list of consecutive ID's $\langle k_0 \rangle, \dots, \langle k_m \rangle$ with $0 \leq k_j \leq Q-1$ and $k_{j+1} = k_j+1$, the ensemble state $\rho_s(0)$ will be spanned in the dimension given by those subsystems, $N_s = \prod_{j=0}^{m} n_{k_j}$ and $\rho_s(0) \in \C^{N_s\times N_s}$ with basis matrices $B^{kj}$ spanned in $\C^{N_s\times N_s}$. The initial state that Quandary propagates is then given by 
\begin{align}
  \rho(0) = \bs{e}_0\bs{e}_0^\dagger \otimes \underbrace{\rho_s(0)}_{\in \C^{N_s\times N_s}} \otimes \, \bs{e}_0 \bs{e}_0^\dagger
\end{align}
where the first $\bs{e}_0$ (before the kronecker product) is the first unit vector in $\R^{\prod_{k=0}^{k_0-1}}$ (i.e. ground state in all preceding subsystems), and the second $\bs{e}_0$ (behind the kronecker products) is the first unit vector in the dimension of all subsequent systems, $\R^{\prod_{k=k_m+1}^{Q-1}}$. 

Note: The ensemble state will be spanned in the \textit{essential} levels of the (sub)system, if applicable, and will then be lifted up to the full dimension by inserting zero rows and columns. 

\subsubsection{Three initial states for gate optimization}\label{subsec:threeinitcond}
\textit{Only valid for the density matrix version, solving Lindblad's master equation.}

When considering gate optimization, it has been shown in \cite{goerz2014optimal} that it is enough to consider only three specific initial states during optimization ($n_{init}=3$), independent of the Hilbert space dimension. Those three initial states are given by
\begin{align}
    \rho(0)_1 &= \sum_{i=0}^{N-1} \frac{2(N-i+1)}{N(N+1)} \bs{e}_i\bs{e}_i^\dagger \\
    \rho(0)_2 &= \sum_{ij=0}^{N-1} \frac{1}{N} \bs{e}_i\bs{e}_j^\dagger\\
    \rho(0)_3 &= \frac{1}{N} I_N
\end{align}
where $I_N\in R^{N\times N}$ is the identity matrix. They are readily implemented in Quandary. Note that it is important to choose the weights $\beta_i, i=1,2,3$ in the objective function appropriately to achieve fast convergence. 

Note: The three initial states are spanned in the \textit{full} dimension of the system, including non-essential levels. The theory for gate optimization with three initial states had been developed for considering \textit{only} essential levels (the gate is defined in the same dimension as the system state evolution), and at this point we are not certain if the theory generalizes to the case when non-essential levels are present. It is advised to optimize on the full basis if non-essential levels are present (or work on the theory, and let us know what you find.). The same holds for $N+1$ initial states below. 

\subsubsection{$N+1$ initial states for gate optimization}
\textit{Only valid for the density matrix version, solving Lindblad's master equation.}


The three initial states from above do not suffice to estimate the fidelity of the realized gate (compare \cite{goerz2014optimal}). Instead, it is suggested in that same paper to choose $N+1$ initial states to compute the fidelity. Those $N+1$ initial states consist of the $N$ diagonal states $B^{kk}$ in the Hilbert space of dimension $N$, as well as the totally rotated state $\rho(0)_2$ from above. Quandary offers the choice to simulate (or optimize) using those initial states, then $n_{init} = N+1$.

Note: The $N+1$ initial states are spanned in the \textit{full} dimension of the system, including non-essential levels, see above for 3-state initialization. 

\subsubsection{Reading an initial state from file}
A specific initial state can also be read from file ($\Rightarrow n_{init}=1$). Format: one column being the vectorized density matrix (vectorization is column-wise), or the state vector, first all real parts, then all imaginary parts (i.e. number of lines is $2N^2$ or $2N$, with one real-valued number per line). 

This option is useful for example if one wants to propagate a specific \textit{non-pure} initial state. In that case, one first has to generate a datafile storing that state (e.g. by simulating a system and storing the output), which can then be read in as initial condition. 



\subsection{Tikhonov regularization, penalty terms, and leakage prevention}\label{sec:penalty}

In order to regularize the optimization problem (stabilize optimization convergence), a standard Tikhonov regularization term can added to the objective function. 
\begin{align}
\mbox{Tikhonov} = \frac{\gamma_1}{2} \| \bfa \|^2_2
\end{align}
By adding this term with a small parameter $\gamma_1 > 0$, the optimization problem will favor optimal control vectors that have a small norm. It regularizes the optimization problem since it adds a small but positive identity matrix to the Hessian of the objective function, hence ``convexifying'' the problem.

In addition to the Tikhonov regularization term, four additional penalty terms can optionally be added to the objective function if desired:
\begin{align*}
  Penalty &= \frac{\gamma_2}{T} \int_0^T P\left(\{\rho_i(t)\}\right) \, \mathrm{d} t   \hspace{3cm} \rightarrow \text{Leakage prevention}\\
         &+  \frac{\gamma_3}{T} \int_0^T \, \| \partial_{tt} \mbox{Pop}(\rho_i(t)) \|^2 \mathrm{d}t \hspace{2cm} \rightarrow \text{State variation penalty} \\
        &+\frac{\gamma_4}{T} \int_0^T \, \sum_k |d^k(\alpha^k,t)|^2\, dt  \hspace{2cm}\rightarrow  \text{Control energy penalty}\\
        &+ \frac{\gamma_5}{2} Var(\vec{\alpha}) \hspace{4cm}\rightarrow  \text{Control variation penalty}
\end{align*}
The first penalty term can be added with $\gamma_2 > 0$ to drive the quantum system towards the desired state over the entire time-domain $0\leq t\leq T$, rather than only at the final time. If extra (non-essential levels are considered through the optimization for at least one oscillator ($n_k^e < n_k$ for at least $k$, compare Sec.~\ref{sec:essentiallevels}), then this term can be used to prevent leakage to higher energy levels that are not modelled. In particular, in that case, the occupation of all \textit{guard levels} are penalized with  
\begin{align}\label{eq:leakprevention}
  P(\rho(t)) = \sum_{r} \| \rho(t)_{rr} \|^2_2
\end{align}
where $r$ iterates over all indices that correspond to a guard level (i.e., the final (highest) non-essential energy level) of at least one of the subsystems, and $\rho(t)_{rr}$ denotes their corresponding population. 

The second penalty term can be added with parameter $\gamma_3 > 0$ to encourage solutions whose populations vary slowly in time by penalizing the second derivative of the populations of the state. 

The third penalty term can be added with parameter $\gamma_4 > 0$ to encourage small control pulse amplitudes by penalizing the control pulse energy. This term can be useful if hardware bounds are given for the control pulse amplitudes: Rather than include amplitude bounds on control pulse directly, which often leads to more non-convex optimization problems and convergence deterioration, one can utilize this penalty term to favor short control pulses with small amplitudes. Compare also \cite{gunther2023practical} for its usage to determine minimal gate durations.

The last penalty term, activated by setting $\gamma_5>0$, is used to penalize variation in control strength between consecutive B-spline coefficients. It is currently only implemented for piecewise zeroth order spline functions, see Section \ref{subsec:bspline-0}, where it is useful to prevent noisy control pulses. Referring to the control function representation in \eqref{eq:spline-ctrl}, this penalty function takes the form:
\begin{align}
  Var(\vec{\alpha}) = \sum_{k=1}^Q Var_k(\vec{\alpha}),\quad Var_k(\vec{\alpha}) = \sum_{f=1}^{N_f^k} \sum_{s=2}^{N_s^k} |\alpha_{s,f}^k - \alpha_{s-1,f}^k|^2,
\end{align}
in terms of the complex-valued control parameters $\alpha_{s,f}^k = \alpha_{s,f}^{k(1)} + i \alpha_{s,f}^{k(2)}$. Penalizing the variance can significantly reduce the noise level in the optimized control functions.

Note: All regularization and penalty coefficients $\gamma_i$ should be chosen small enough so that they do not dominate the final-time objective function $J$. This might require some fine-tuning. It is recommended to always add $\gamma_1>0$, e.g. $\gamma_1 = 10^{-4}$, and add other penalties only if needed. 


% Achieving a target at EARLIER time-steps:
% \begin{align}\label{eq:penaltyterm}
%   P(\rho(t))  =  w(t) J\left(\rho(t)\right) \quad \text{where} \quad w(t) =
%   \frac{1}{a} e^{ -\left(\frac{t-T}{a} \right)^2},
% \end{align}
% for a penalty parameter $0 \leq a \leq 1$. Note, that as $a\to 0$, the weighting function $w(t)$ converges to the Dirac delta distribution with peak at final time $T$, hence reducing $a$ leads to more emphasis on the final time $T$ while larger $a$ penalize non-zero energy states at earlier times $t\leq T$.  

\section{Implementation}

  \subsection{Vectorization of Lindblad's master equation}
  When solving Lindblad's master equation \eqref{mastereq}, Quandary uses a vectorized representation of the density matrix with $q(t) := \text{vec}(\rho(t)) \in \C^{N^2}$ (column-wise vectorization). Using the
  relations
  \begin{align}
   \text{vec}(AB) &= (I_N\otimes A)\text{vec}(B) = (B^T\otimes I_N)\text{vec}(A)
    \\
   \text{vec}(ABC) &= (C^T\otimes A)\text{vec}(B)
  \end{align}
  for square matrices $A,B,C\in\C^{N\times N}$, the vectorized
  form of the Lindblad master equation is given by:
  \begin{align}\label{mastereq_vectorized}
    &\dot q(t) = M(t) q(t) \quad  \text{where} \\
    &M(t) := -i(I_N\otimes H(t) - H(t)^T \otimes I_N) + \sum_{k=0}^{Q-1}\sum_{l=1}^2 \gamma_{lk}
    \left( \Ell_{lk}\otimes \Ell_{lk} - \frac 1 2 \left( I_N\otimes
    \Ell^T_{lk}\Ell_{lk} + \Ell^T_{lk}\Ell_{lk} \otimes I_N \right) \right)
  \end{align}
   with $M(t) \in \C^{N^2\times N^2}$, and $H(t) = H_d(t) + H_c(t)$ being the rotating frame system and control Hamiltonians as in \eqref{eq:Hd_rotating} and \eqref{eq:Hc_rotating}, respectively.

   When solving Schroedinger's equation \eqref{eq:schroedinger}, Quandary operates directly on the state $q(t) := \psi(t)\in\C^N$ and solves \eqref{mastereq_vectorized} with $M(T) := -iH(t)$.
    
  \subsubsection{Real-valued system and state storage}
   Quandary solves the (vectorized) equation \eqref{mastereq_vectorized} in
   real-valued variables with $q(t) = u(t) + iv(t)$, evolving the real-valued
   states $u(t), v(t)\in \R^{M}$ for $M=N$ (Schroedinger's eq.) or $M=N^2$ (Lindblad's eq.) with
   \begin{align}
     \dot q(t) = M(t) q(t) \quad \Leftrightarrow \quad \begin{bmatrix} \dot u(t) \\ \dot v(t) \end{bmatrix} = 
   \begin{bmatrix} A(t) & -B(t) \\ B(t) & A(t) \end{bmatrix} 
   \begin{pmatrix} u(t) \\ v(t) \end{pmatrix} 
   \label{realvaluedODE}
   \end{align}
   for real and imaginary parts $A(t) = \mbox{Re} \left(M(t)\right)$ and $B(t) = \mbox{Im}\left(M(t)\right)$. 

The real and imaginary parts of $q(t)$ are stored in a colocated manner: For
  $q = u+iv$ with $u,v\in\R^{M}$, a vector of size $2M$ is stored that
  staggers real and imaginary parts behind each other for each component:
  \begin{align*}
    q = u+iv = \begin{bmatrix}
     u^1\\u^2\\ \vdots \\ u^{M} 
    \end{bmatrix}
    + i \begin{bmatrix}
     v^1\\v^2\\ \vdots \\ v^{M} 
    \end{bmatrix}
    \quad \Rightarrow \quad
    q_{store} = \begin{bmatrix}
      u_1 \\ v_1\\ u_2 \\ v_2 \\ \vdots \\ u_{M} \\ v_{M}
    \end{bmatrix}
  \end{align*}


 \subsection{Sparse-matrix vs. matrix-free solver}

   In Quandary, two versions to evaluate the right hand side of Lindblad's
   equation, $M(t)q(t)$, of the vectorized real-valued system are available: 
   \begin{enumerate}
     \item The \textit{sparse-matrix solver} uses PETSc's sparse matrix format (sparse AIJ) to set up (and store) the time-independent building blocks inside $A(t)$ and $B(t)$. Sparse matrix-vector products are then applied at each time-step to evaluate the products $A(t)u(t) - B(t) v(t)$ and $B(t)u(t) + A(t)v(t)$.  
     
     For developers, the appendix provides details on each term within $A(t)$ and $B(t)$ which can be matched to the implementation in the code (class \texttt{MasterEq}). 

     \item The \textit{matrix-free solver} considers the state density matrix $\rho\in C^{N\times N}$ to be a tensor of rank $2Q$ (one axis for each subsystems for each matrix dimension, hence $2\cdot Q$ axes). Instead of storing the matrices within $M(t)$, the matrix-free solver applies tensor contractions to realize the action of $A(t)$ and $B(t)$ on the state vectors. 

    In our current test cases, the matrix-free solver is much faster than the sparse-matrix solver (about 10x), no surprise. However the matrix-free solver is currently only implemented for composite systems consisting of \textbf{2, 3, 4, or 5} subsystems. 

    \textbf{The matrix-free solver currently does not parallelize across the system dimension $N$}, hence the state vector is \textbf{not} distributed (i.e. no parallel Petsc!). The reason why we did not implement that yet is that $Q$ can often be large while each axis can be very short (e.g. modelling $Q=12$ qubits with $n_k=2$ energy levels per qubit), which yields a very high-dimensional tensor with very short axes. In that case, the standard (?) approach of parallelizing the tensor along its axes will likely lead to very poor scalability due to high communication overhead. We have not found a satisfying solution yet - if you have ideas, please reach out, we are happy to collaborate! 
   \end{enumerate} 


    \subsection{Time-stepping}
    To solve the (vectorized) master equation \eqref{mastereq_vectorized}, $\dot
    q(t) = M(t) q(t)$ for $t\in [0,T]$, Quandary applies a time-stepping integration
    scheme on a uniform time discretization grid $0=t_0 < \dots t_{N} = T$, with
    $t_n = n \delta t$ and $\delta t = \frac{T}{N}$, and approximates the
    solution at each discrete time-step $q^{n} \approx q(t_n)$. The time-stepping scheme can be chosen in Quandary through the configuration option \texttt{timestepper}. 
    
    \subsubsection{Implicit Midpoint Rule (\texttt{IMR})}
    The implicit midpoint rule is a second-order accurate, symplectic time-stepping algorithm with Runge-Kutta scheme tableau 
    \begin{tabular}{ c | c }
      $1/2$ & $ 1/2$ \\
      \hline
                &  $1$
    \end{tabular}.
    Given a state $q^n$ at time $t_n$, the update formula to compute $q^{n+1}$
    is hence 
    \begin{align}
      q^{n+1} = q^n + \delta t k_1 \quad \text{where} \, k_1 \, \text{solves}
      \quad \left( I-\frac{\delta t}{2} M^{n+1/2} \right) k_1 = M^{n+1/2}  q^n
    \end{align}
    where $M^{n+1/2} := M(t_n + \frac{\delta t}{2})$. In each time-step,
    a linear equation is solved to get the stage variable $k_1$, which is then used it
    to update $q^{n+1}$. 

    \subsubsection{Higher-order compositional IMR (\texttt{IMR4}, or \texttt{IMR8})}
    A compositional version of the Implicit Midpoint Rule is available that performs multiple IMR steps in each time-step interval, which are composed in such a way that the resulting compositional step is of higher order. Currently, Compared to the standard IMR, the higher-order methods can be very beneficial as it allows for much larger time-steps to be taken to reach a certain accuracy tolerance. Even though more work is done per time-step, the reduction in the number of time-steps needed can be several orders or magnitude and there is hence a tradeoff where the compositional methods outperform the standard IMR scheme.

    Currently available is a compositional method of 4-th order that performs 3 sub-steps per time-step (\texttt{IMR4}), and a compositional method of 8-th order performing 15 sub-steps per time-step (\texttt{IMR8}).

    \subsubsection{Choice of the time-step size}
  The python interface to Quandary automatically computes a time-step size based on the fastest period of the system Hamiltonian. For the C++ code, it needs to be set by the user.

    In order to choose a time-step size $\delta t$, an eigenvalue analysis of
    the constant drift Hamiltonian $H_d$ is often useful. Since $H_d$ is Hermitian, there exists a transformation $Y$ such that $Y^{\dagger}H_d Y = \Lambda \qquad  \text{where} \quad Y^{\dagger} = Y$ where $\Lambda$ is a diagonal matrix containing the eigenvalues of $H_d$.
       Transform the state $\tilde q = Y^{\dagger} q$, then the ODE transforms to 
       \begin{align*}
         \dot \tilde q = -i \Lambda \tilde q \quad \Rightarrow \dot \tilde q_i =
         -i\lambda_i \tilde q_i \quad \Rightarrow \tilde q_i = a
         \exp(-i\lambda_i t)
       \end{align*}
       Therefore, the period for each mode is $\tau_i =
       \frac{2\pi}{|\lambda_i|}$, hence the shortest period is $\tau_{min} =
       \frac{2\pi}{\max_i\{|\lambda_i|\}}$. If we want $p$ discrete time points
       per period, then $p\delta t = \tau_{min}$, hence 
       \begin{align} \label{eq:timestepsize}
         \delta t = \frac{\tau_{min}}{p} = \frac{2\pi}{p\max_i\{|\lambda_i|\}}
       \end{align}
       Usually, for the second order scheme we would use something like $p=40$. The above estimate provides a first idea on how big (small) the time-step size should be, and the user is advised to consider this estimate when running a test case. However, the estimate ignores contributions from the control Hamiltonian, where larger control amplitudes will require smaller and smaller time-steps in order to resolve (a) the time-varying controls themselves and (b) the dynamics induced by large control contributions. A standard $\Delta t$ test should be performed in order to verify that the time-step is small enough. For example, one can compute the Richardson error estimator of the current approximation error to some true quantity $J^*$ from
       \begin{align}
         J^* - J^{\Delta t} = \frac{J^{\Delta t} - J^{\Delta t m}}{1-m^p} + O(\Delta t^{p+1})
       \end{align}
       where $p$ is the order of the time-stepping scheme (i.e. $p=2$ for the IMR and $p=8$ for the compositional IMR8), and $J^{\Delta t}, J^{\Delta tm}$ denote approximations thereof using the time-stepping sizes $\Delta t$ and $\Delta t m$ for some factor $m$.        
       
      %  If one wants to include the time-varying Hamiltonian part $H = H_d +
      %  H_c(t)$ in the analysis, one could use the constraints on the control
      %  parameter amplitudes to remove the time-dependency using their large
      %  value instead.       

  
  \subsection{Gradient computation via discrete adjoint back-propagation}
   Quandary computes the gradients of the objective function with respect to the design variables $\boldsymbol{\alpha}$ using the discrete adjoint method. The discrete adjoint approach yields exact and consistent gradients on the algorithmic level, at costs that are independent of the number of design variables.    
   To that end, the adjoint approach propagates local sensitivities backwards through the time-domain while concatenating contributions to the gradient using the chain-rule.

  The consistent discrete adjoint time-integration step for
    adjoint variables denoted by $\bar q^{n}$ is given by
   \begin{align}
      \bar q^{n} = \bar q^{n+1} + \delta t \left(M^{n+1/2}\right)^T \bar k_1
      \quad \text{where} \, \bar k_1 \, \text{solves} \quad \left(
      I-\frac{\delta t}{2} M^{n+1/2}\right)^T  \bar k_1 = \bar q^{n+1} 
    \end{align}
  The contribution to the gradient $\nabla J$ for each time-step is
    \begin{align}\label{eq:gradient}
      \nabla J += \delta t \left( \frac{\partial M^{n+1/2}}{\partial z}
      \left(q^n + \frac{\delta t}{2} k_1\right) \right)^T\bar k_1
    \end{align}
  
    Each evaluation of the gradient $\nabla J$ involves a forward solve of $n_{init}$ initial quantum states to evaluate the objective function at final time $T$, as well as $n_{init}$ backward solves to compute the adjoint states and the contributions to the gradient. Note that the gradient computation \eqref{eq:gradient} requires the states and adjoint states at each time-step. For the Schroedinger solver, the primal states are recomputed by integrating Schroedinger's equation backwards in time, alongside the adjoint computation. For the Lindblad solver, the states $q^n$ are stored during forward propagation, and taken from storage during adjoint back-propagation (since we can't recompute it in case of Lindblad solver, due to dissipation). 


  \subsection{Optimization algorithm}
    Quandary utilized Petsc's \texttt{Tao} optimization package to apply gradient-based iterative updates to the control variables. The \texttt{Tao} optimization interface takes routines to evaluate the objective function as well as the gradient computation. In the current setting in Quandary, \texttt{Tao} applies a nonlinear Quasi-Newton optimization scheme using a preconditioned gradient based on L-BFGS updates to approximate the Hessian of the objective function. A projected line-search is applied to ensure that the objective function yields sufficient decrease per optimization iteration while keeping the control parameters within the prescribed box-constraints. 


    \section{Parallelization}
    Quandary offers two levels of parallelization using MPI. 
    \begin{enumerate}
    \item Parallelization over initial conditions: The $n_{init}$ initial conditions $\rho_i(0)$ can be distributed over \texttt{np\_init} compute units. Since initial condition are propagated through the time-domain for solving Lindblad's or Schroedinger's equation independently from each other, speedup from distributed initial conditions is ideal. 
    \item Parallel linear algebra with Petsc (sparse-matrix solver only): For the sparse-matrix solver, Quandary utilizes Petsc's parallel sparse matrix and vector storage to distribute the state vector onto \texttt{np\_petsc} compute units (spatial parallelization). To perform scaling results, make sure to disable code output (or reduce the output frequency to print only the last time-step), because writing the data files invokes additional MPI calls to gather data on the master node.
    \end{enumerate}
    Strong and weak scaling studies are presented in \cite{guenther2021quantum}.

    Since those two levels of parallelism are orthogonal, Quandary splits the global communicator (MPI\_COMM\_WORLD) into
    two sub-communicator such that the total number of executing MPI
    processes ($np_{total}$) is split as
    \begin{align*}
      np_{init} * np_{petsc} = np_{total}.
    \end{align*}
    Since parallelization over different initial conditions is perfect, Quandary automatically sets $np_{init} = n_{init}$, i.e. the total number of cores for distributing initial conditions is the total number of initial conditions that are considered in this run, as specified by the configuration option \texttt{intialcondition}. The number of cores for distributed linear algebra with Petsc is then computed from the above equation.

    It is currently required that the number of total cores for executing quandary is an integer divisor of multiplier of the number of initial conditions, such that each processor group owns the same number of initial conditions. 
    
    It is further required that the system dimension is an integer multiple of the number of cores used for distributed linear algebra from Petsc, i.e. it is required that $\frac{M}{np_{petsc}} \in \mathds{N}$ where $M=N^2$ in the Lindblad solver case and $M=N$ in the Schroedinger case. This requirement is a little
      annoying, however the current implementation requires this due to the
      colocated storage of the real and imaginary parts of the vectorized
      state.
 
\section{Output and plotting the results}
Quandary generates various output files for system evolution of the current (optimized) controls as well as the optimization progress. All data files will be dumped into a user-specified folder through the config option \texttt{datadir}. 

\subsection{Output options with regard to state evolution}
For each subsystem $k$, the user can specify the desired state evolution output through the config option \texttt{output<k>}:
\begin{itemize}
  \item \texttt{expectedEnergy}: This option prints the time evolution of the expected energy level of subsystem $k$ into files with naming convention \texttt{expected<k>.iinit<m>.dat}, where $m=1,\dots,n_{init}$ denotes the unique identifier for each initial condition $\rho_m(0)$ that was propagated through (see Section \ref{subsec:initcond}). This file contains two columns, the first row being the time values, the second one being the expectation value of the energy level of subsystem $k$ at that time point, computed from 
  \begin{align}
    \langle N^{(n_k)}\rangle = \mbox{Tr}\left(N^{(n_k)} \rho^k(t)\right)
  \end{align}
  where $N^{(n_k)} = \left(a^{(n_k)}\right)^\dagger \left(a^{(n_k)}\right)$ denotes the number operator in subsystem $k$ and $\rho^k$ denotes the reduced density matrix for subsystem $k$, each with dimension $n_k\times n_k$. Note that this equivalent to $\mbox{Tr}\left(N_k \rho(t)\right)$ with $N_k = I_{n_1} \otimes \dots \otimes I_{n_{k-1}} \otimes N^{(n_k)} \otimes I_{n_{k+1}}\otimes \dots \otimes I_Q$ and the full state $\rho(t)$ in the full dimensions $N\times N$.
  \item \texttt{expectedEnergyComposite} Prints the time evolution of the expected energy level of the entire (full-dimensional) system state into files (one for each initial condition, as above): $mbox{Tr}\left(N \rho(t)\right)$ for the number operator $N$ in the full dimensions. 
  \item \texttt{population}: This option prints the time evolution of the state populations (diagonal of density matrix, state probabilities) of subsystem $k$ into files named \texttt{population<k>.iinit<m>.dat} for each initial condition $m=1,\dots, n_{init}$. The files contain $n_k+1$ columns, the first one being the time values, the remaining $n_k$ columns correspond to the population of each level $l=0,\dots,n_k-1$ of the reduced density matrix $\rho^k(t)$ at that time point. For Lindblad's solver, these are the diagonal elements of the reduced density matrix ($\rho_{ll}^k(t), l=0,\dots n_k-1$), for Schroedinger's solver it's the absolute values of the reduced state vector elements $|\psi^k_l(t)|^2, l=0,\dots n_k-1$. Note that the reduction to the subsystem $k$ induces a sum over all oscillators to collect contributions to the reduced state. 
  \item \texttt{populationComposite}: Prints the time evolution of the state populations of the entire (full-dimensional) system into files (one for each initial condition, as above). 
  \item \texttt{fullstate}: Probably only relevant for debugging or very small systems, one can print out the full state $\rho(t)$ or $\psi(t)$ for each time point into the files \texttt{rho\_Re.iinit<m>.dat} and \texttt{rho\_Im.iinit<m>.dat}, for the real and imaginary parts of the state, respectively. These files contain $N^2 +1$ (Lindblad) or $N+1$ (Schroedinger) columns the first one being the time point value and the remaining ones contain the vectorized density matrix (Lindblad, $N^2$ elements) or the state vector (Schroedinger, $N$ elements) for each time-step. Note that these file become very big very quickly -- use with care!
\end{itemize}

The user can change the frequency of output in time (printing only every $j$-th time point) through the option \texttt{output\_frequency}. This is particularly important when doing performance tests, as computing the reduced states for output requires extra computation and communication that might skew performance tests. 

\subsection{Output with regard to simulation and optimization}
\begin{itemize}
  \item \texttt{config\_log.dat} contains all configuration options that had been used for the current run. 
  \item \texttt{params.dat} contains the control parameters $\bfa$ that had been used to determine the current control pulses. This file contains one column containing all parameters, ordered as stored, see Section \ref{subsec:controlpulses}.
  \item \texttt{control<k>.dat} contain the resulting control pulses applied to subsystem $k$ over time. It contains four columns, the first one being the time, second and third being $p^k(t)$ and $q^k(t)$ (rotating frame controls), and the last one is the corresponding lab-frame pulse $f^k(t)$. Note that the units of the control pulses are in frequency domain (divided by $2\pi)$. The unit matches the unit specified with the system parameters such as the qubit ground frequencies $\omega_k$.
  \item \texttt{optim\_history.dat} contains information about the optimization progress in terms of the overall objective function and contribution from each term (cost at final time $T$ and contribution from the tikhonov regularization and the penalty term), as well the norm of the gradient and the fidelity, for each iteration of the optimization. If only a forward simulation is performed, this file still prints out the objective function and fidelity for the forward simulation. 
\end{itemize}
Quandary always prints the current parameters and control pulses at the beginning of a simulation or optimization, and in addition at every $l$-th optimization iteration determined from the \texttt{optim\_monitor\_frequency} configuration option. 

\subsection{Plotting}
The format of all output files are very well suited for plotting with \href{http://www.gnuplot.info}{Gnuplot}, which is a command-line based plotting program that can output directly to screen, or into many other formats such as png, eps, or even tex. As an example, from within a Gnuplot session, you can plot e.g. the expected energy level of subsystem $k=0$ for initial condition $m=0$ by the simple command\newline
\texttt{gnuplot> plot 'expected0.iinit0000.dat' using 1:2 with lines title 'expected energy subsystem 0'} 
\\
which plots the first against the second column of the file 'expected0.iinit0000.dat' to screen, connecting each point with a line. Additional lines (and files) can be added to the same plot by extending the above command with another file separated by comma (only omit the 'plot' keyword for the second command). There are many example scripts for plotting with gnuplot online, and as a starting point I recommend looking into some scripts in the 'quandary/util/' folder.

\section{Testing}
  \begin{itemize}
    \item Quandary has a set of regression tests. Please take a look at the \verb+tests/regression/README.md+ document for instructions on how to run the regression tests.
    \item In order to check if the gradient implementation is correct, one can choose to run a Central Finite Difference test. Let the overall objective function be denoted by $F(\boldsymbol{\alpha})$. The Central Finite Difference test compares each element of the gradient $\nabla F(\boldsymbol{\alpha})$ with the following (second-order accurate) estimate:
    \begin{align*}
       \left(\nabla F(\boldsymbol{\alpha}) \right)_i \approx \frac{F(\bfa + \epsilon\bs{e}_i) - F(\bfa - \epsilon\bs{e}_i)}{2\epsilon} \qquad \qquad \text{(CFD)}
    \end{align*}
    for unit vectors $\bs{e}_i\in \R^d$, and $d$ being the dimension of $\bfa$. 

    To enable the test, set the flag for the compiler directive \texttt{TEST\_FD\_GRAD} at the beginning of the \texttt{src/main.cpp} file. Quandary will then iterate over all elements in $\alpha$ and report the \textit{relative} error of the implemented gradient with respect to the ``true'' gradient computed from CFD. 

  \end{itemize}

\section*{Acknowledgments}
This work was performed under the auspices of the U.S. Department of Energy by Lawrence
Livermore National Laboratory under Contract DE-AC52-07NA27344. LLNL-SM-818073. 

% This document was prepared as an account of work sponsored by an agency of the United States
% government. Neither the United States government nor Lawrence Livermore National Security, LLC,
% nor any of their employees makes any warranty, expressed or implied, or assumes any legal
% liability or responsibility for the accuracy, completeness, or usefulness of any information,
% apparatus, product, or process disclosed, or represents that its use would not infringe
% privately owned rights. Reference herein to any specific commercial product, process, or service
% by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply
% its endorsement, recommendation, or favoring by the United States government or Lawrence
% Livermore National Security, LLC. The views and opinions of authors expressed herein do not
% necessarily state or reflect those of the United States government or Lawrence Livermore
% National Security, LLC, and shall not be used for advertising or product endorsement purposes.


\bibliographystyle{plain}
\bibliography{user_guide}


\appendix
 \section{Appendix: Details for the real-valued, vectorized Hamiltonian}
   To assemble (evaluate) 
   $A(t) = Re(M(t))$ and $B(t) = Im(M(t))$, consider
   \begin{align}
    iH &= iH_d(t) + iH_c(t) \\
      &= i\left( \sum_k (\omega_k - \omega_k^{\text{rot}}) a_k^\dagger a_k - \frac{\xi}{2}a_k^\dagger a_k^\dagger a_k a_k  - \sum_{l>k}  \xi_{kl} a_k^\dagger a_k a_l^\dagger a_l +  \sum_{l>k} J_{kl} \cos(\eta_{kl}t)\left(a_k^\dagger a_l + a_ka_l^\dagger \right)
       \right.\\
      & \left. \quad \quad + \sum_k p^k(\vec{\alpha}^k,t) \left(a_k + a_k^\dagger\right) \right)\\
      &+ \left( \sum_k \sum_{kl} - J_{kl} \sin(\eta_{kl}t) \left(a_k^\dagger a_l - a_ka_l^\dagger\right)  - \sum_k q^k(\vec{\alpha}^k, t)\left(a_k - a_k^\dagger\right)\right)
   \end{align}
   Hence $A(t)$ and $B(t)$ are given by 
   \begin{align}
    A(t) &= A_d + \sum_k  q^k(\vec{\alpha}^k,t) A_c^k + \sum_{l>k} J_{kl} \sin(\eta_{kl}t)  A_d^{kl} \\
   \text{with} \quad  A_d &:= \sum_k \sum_{j=1,2} \gamma_{jk} \left( \Ell_{jk}\otimes\Ell_{jk} - \frac 1 2 \left(I_N \otimes \Ell_{jk}^T\Ell_{jk} + \Ell_{jk}^T\Ell_{jk}\otimes I_N\right) \right)\\
    A_c^k &:=  I_N \otimes \left(a_k - a_k^\dagger\right) - \left(a_k - a_k^\dagger\right)^T\otimes I_N \\
    A_d^{kl} &:=  I_N\otimes \left(a_k^\dagger a_l - a_k a_l^\dagger\right) - \left(a_k^\dagger a_l - a_k a_l^\dagger\right)^T\otimes I_N 
   \end{align}
   and
   \begin{align}
     B(t) &=  B_d + \sum_k p^k(\vec{\alpha}^k,t) B_c^k + \sum_{kl} J_{kl} \cos(\eta_{kl}t)B_d^{kl}\\
     \text{with} \quad B_d &:= \sum_k (\omega_k - \omega_k^{\text{rot}}) \left(-I_N \otimes a_k^\dagger a_k + (a_k^\dagger a_k)^T \otimes I_N \right) - \frac{\xi_k}{2}\left(- I_N \otimes a_k^\dagger a_k^\dagger a_k a_k + (a_k^\dagger a_k^\dagger a_k a_k )^T\otimes I_N\right)  \\
       &\quad - \sum_{l>k}  \xi_{kl} \left(-I_N \otimes a_k^\dagger a_k a_l^\dagger a_l + (a_k^\dagger a_k a_l^\dagger a_l)^T \otimes I_N \right)\\
       B_c^k &:=  - I_N \otimes \left(a_k + a_k^\dagger\right) + \left(a_k + a_k^\dagger\right)^T\otimes I_N \\
       B_d^{kl} &:=  - I_N\otimes \left(a_k^\dagger a_l + a_k a_l^\dagger\right) + \left(a_k^\dagger a_l + a_k a_l^\dagger\right)^T\otimes I_N \\
   \end{align}

  
  The sparse-matrix solver initializes and stores the constant matrices
       $A_d, A_d^{kl}, A_c^k, B_d, B_d^{kl}, B_c^k$ using Petsc's sparse-matrix format. They are used
       as building blocks to evaluate the blocks in the system matrix $M(t)$ with 
     \begin{align}
       A(t) &= Re(M(t)) = A_d + \sum_k q^k(\alpha^k, t)A_c^k + \sum_{l>k} J_{kl} \sin(\eta_{kl}t) A_d^{kl}\\
       B(t) &= Im(M(t)) = B_d + \sum_k p^k(\alpha^k, t)B_c^k + \sum_{kl} J_{kl} \cos(\eta_{kl}t) B_d^{kl}
     \end{align}
   at each time $t$, which are applied to the vectorized, real-valued density matrix using Petsc's sparse MatVec implementation. 

  The matrix-free solver does not explicitly store the matrices $A_d,B_d,
       A_c^k, B_c^k$, etc., but instead only evaluates their action on a vector $q(t)$ using tensor contractions applied to the corresponding dimension of the density matrix tensor. 
  



  \section{Summary of all C++ configuration options}
  Here is a list of all options available to the C++ Quandary code, this is the same as in \texttt{config\_template.cfg}.
 
  \lstinputlisting[breaklines]{../../config_template.cfg}


  \section{Summary of all python interface options}
  Here is a list of all options available to the python interface, this is the same as in \texttt{quandary.py}.

  \lstinputlisting[breaklines, firstline=17, lastline=91, keepspaces=false, language=mylanguage]{../../quandary.py}
  

\end{document}
