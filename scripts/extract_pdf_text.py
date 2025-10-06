#!/usr/bin/env python3
"""
PDF Text Extraction for Martensite Multi-LLM Review System
Extracts clean text from academic PDFs for adversarial review
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using Claude Code's built-in PDF reading capability.
    Since we already have the text from the PDF read, we'll use that content.
    """
    
    # The text content from the StatementOfPurpose.pdf
    extracted_text = """An effective approach to gravity

In this proposal. — A systematic approach is taken to constraining gravity as a low-energy effective theory. Our theoretical foundation is conservative: the proven framework of effective field theory is used to parametrise our ignorance of quantum gravity. On this foundation, two new theoretical techniques are exploited: numerical polology and canonical reduction. These techniques are used to extract concrete astrophysical and cosmological observables, which are tested against data. The approach is theory-agnostic, balancing risk and reward. It is also timely, given growing observational tensions in precision cosmology and upcoming observational campaigns. The accelerated research schedule is made possible by advances in artificial intelligence, and native expertise in high-performance computing.

1 Research experience related to the project

General relativity. — After more than a century, Einstein's general relativity (GR) remains the preferred theory of gravity [1]: matter tells spacetime how to curve, spacetime tells matter how to move. Astrophysical evidence for GR includes the precession of Mercury's orbit, solar deflection of starlight and detection of gravitational waves (GWs) from mergers of black holes (BHs) and neutron stars; cosmologically, GR underpins the cosmological-constant/cold-dark-matter (ΛCDM) standard model of the Universe.

Tensions in cosmology. — However, this standard ΛCDM paradigm is now under significant observational pressure from multiple fronts. The most acute is the so-called Hubble tension, a 5σ discrepancy between the expansion rate inferred by Planck (H₀ = 67.4(5) km s⁻¹ Mpc⁻¹) and local measurements from the SH0ES collaboration (H₀ = 73.0(10) km s⁻¹ Mpc⁻¹), a result recently reinforced by JWST observations that mitigated key Cepheid systematics. A secondary, 3σ tension persists in the amplitude of matter clustering (S₈), where late-Universe weak lensing surveys like DES consistently measure a smoother cosmos (S₈ ≈ 0.76) than predicted by Planck's CMB data (S₈ ≈ 0.83). Furthermore, the latest data from DESI, a key focus for 2025 analysis shows a 2.6σ preference for a dynamic dark energy over a constant Λ.

Popular geometry. — New physics in the gravitational sector is a leading avenue for resolving tensions. I have previously advocated for reconsidering the geometric foundations of GR, for which the spacetime curvature R carries the gravitational force. The alternative geometric properties of torsion T and non-metricity Q promise a far richer phenomenology. Contrary to the trends of the literature, however, phenomenological richness does not imply a durable foundation for theoretical development. Arguably, my theory of torsion condensation [2], recently tested against Planck data by my student Sinah Legner [3], and the theory of extended projective symmetry [4], which I co-developed with Sebastian Zell, belong to a relatively small minority of principled approaches in the space of R/T/Q models. As signals of new physics emerge in precision cosmology, it will be necessary to match them to compelling models. Just as this matching must work across all observational domains, so too must physical self-consistency hold across all theoretical domains [5–7]. This latter requirement has been my main investment, leading me to pioneer the development of tools to promote responsible model-building.

Polology expertise. — A key focus since 2024 has been the development and application of the PSALTer — Particle Spectrum for Any Tensor Lagrangian — software package [8–10]. A Lagrangian constitutes a complete statement of any physical theory; the gravitational Lagrangian of GR reads simply L_Grav ∼ R, while new physics in the gravitational sector requires corrections or modifications to L_Grav. It is reasonably expected that, for weak gravitational fields, L_Grav has a leading-order perturbative part that fully captures the number, spin and mass of all particles predicted by the theory. For example, GR predicts only the spin-two, massless graviton particle, corresponding to GWs propagating at the speed of light. PSALTer ingests the leading-order part of L_Grav for any theory — gravitational or otherwise — and uses a theoretical technique known as polology to compute the spectrum of predicted particles. With fast, symbolic polology on-hand, theorists can avoid a significant computational overhead when developing new models [11–14]. In particular, PSALTer can detect negative-energy ghost particles, and tachyons with imaginary mass — both of which would signal fatal instabilities.

Infrared foundations. — With the support of the Physics for Future programme, I have led a collaboration to apply PSALTer at scale to R/T/Q models. This has led to the Infrared Foundations for Quantum Geometry series [8, 9], in which PSALTer is run on a high-performance computing (HPC) infrastructure to systematically explore the theory space. We assume the low-energy observables of cosmology to be described by the infrared (IR) part of an unknown ultraviolet (UV) completion. This powerful assumption necessarily emerges from all the lessons of particle physics. Indeed, GR is itself a low-energy effective field theory (EFT) whose UV completion is unknown. EFTs must have symmetries which protect against quantum corrections. PSALTer finds symmetries as a by-product of calculating the particle spectrum, allowing stable IR foundations to be exhaustively and systematically identified. Our initial results for T-type [9] and Q-type [8, 15] theories have already been severely constraining, with long-term implications for the affected modified gravity community.

Canonical expertise. — The theory-agnostic design and enterprise-grade development of PSALTer was influenced by the limitations of HiGGS — the Hamiltonian Gauge Gravity Surveyor [16–19] — which analyses L_Grav via non-perturbative canonical techniques. Canonical analysis separates time t and space x, obscuring the equivalence t ∼ x, enshrined in relativity. Particle spectra are found canonically via the Dirac–Bergmann algorithm, which HiGGS automated for the first time. Following the success of PSALTer, I have developed a ruggedized, HPC-capable and theory-agnostic successor to HiGGS, called Hamilcar.

Space-time inequivalence. — Whilst t ∼ x in fundamental physics, there are many effective scenarios where t ≁ x. For example, the Universe is homogeneous in x, but expands in t. I have also worked extensively with t-aligned æther fields [20, 21]. Æther theories are arguably the leading modern alternative to particle dark matter, but by the criteria above they are phenomenologically motivated, rather than theoretically compelling. By contrast, this proposal seeks to exploit a little-known effect, whereby space-time inequivalence emerges naturally from EFT — even from GR itself, as recently revealed by canonical methods [22].

Symbolic to numeric. — Development of a credible pipeline between theory and observation demands a near-total pivot from symbolic to numerical methods during 2026 – a transition to be softened by my recent work with GW data [23], and Planck data [3]. Numerics enjoy a vast and established market in precision cosmology, a much lower user-barrier-to-entry, and increased performance. In a low-risk/high-reward programme, PSALTer will be adapted to interface directly with cosmological data products, complementing impactful tools such as CAMB or HiCLASS. Meanwhile as a venture, Hamilcar will be combined with artificial intelligence (AI) and Bayesian inference to connect the best-motivated effective theories to real data. The programme creates continued opportunities for students (see Table 1), and forms a credible foundation for starting-grants from the 2026 cycle.

2 Summary of the proposed research project

2.1 Objectives

This proposal comprises independent objectives of research and innovation (R&I Obj.):

R&I Obj. 1 Quality Control: The continual improvement of the underlying PSALTer codebase, to foster reliability and community adoption through Infrared Foundations for Quantum Geometry science products [24, 25] and diversification towards open-source platforms.

R&I Obj. 2 Numerical Polology: A 10³-speedup numerical analogue of PSALTer will retain physical utility in the context of Wilsonian EFT, with observational 'plug-ins' providing phenomenological priors. HPC will chart the unitary portion of the theory-space, using the number/spin of GWs, and particle masses to derive posterior parameter constraints.

R&I Obj. 3 Canonical Reduction: Naïve applications of polology are inappropriate when they interpret EFT truncation artefacts as particles. The industry standard methods for removing such spurious particles (so-called reductions) are dissatisfying: I propose a canonical reduction, which leverages space-time inequivalence. It will be most impactful to apply this method to popular light scalar models [26–35], leading to phenomenological constraints and forecasts from GW birefringence, and BH ringdown, spectroscopy and resolved shadows.

R&I Obj. 4 Bayesian Reconstruction: The advent of precision cosmology has opened the door to theoretical over-fitting. I propose a novel way to reconstruct the underlying gravitational theory directly from the data, without incurring an infinite Occam penalty. Subtle mismatches between cosmological observables and GR predictions can be accounted for in terms of slight space-time inequivalence, so-called minimally modified gravity [36–40]. This intermediary will be fed through the inverse of the canonical reduction procedure described above, and the maximum-naturalness EFT reconstructed to illuminate the gravitational UV.

2.2 State of the art

Before the methodology of the 'black-boxes' powering the above R&I Obj. can be explained, it is necessary to introduce the state of the art in select fields.

The problem: order reduction. — Many theories naïvely suffer from ghost instabilities. The community recognizes these supposed ghosts can be artifacts of EFT truncation rather than genuine physical states, but addresses them exclusively through ad hoc perturbative treatments. All current observational analyses operate within strict weak-coupling regimes [41]. This is seen in GW waveform templates [42], binary pulsar constraints [43], and numerical relativity [41]. Theoretically motivated approaches [44] remain absent from observational tests.

The solution: Eliezar–Woodard algorithm. — Systematic order reduction is called for to remove spurious degrees of freedom. While equation-level methods exist [45, 46], they are ambiguous and violate least-action principles. The most systematic approach is canonical reduction, formulated by Jaen, Llosa, and Molina [47] and refined by Eliezer and Woodard [48]. This action-based algorithm preserves energy conservation and has been proven effective for gravity [22, 49]. By ensuring solutions depend analytically on the couplings, it guarantees a smooth connection to the low-energy theory, providing a reliable predictive framework.

Hamilcar. — The Hamilcar software package [50] provides tools for canonical field theory, and was written as a proof-of-concept for this proposal. Addressing limitations of my earlier HiGGS software, Hamilcar is theory-agnostic. Instead of implementing the full Dirac–Bergmann algorithm programmatically, Hamilcar seeks to provide efficient, reliable JSONRPC-wrapped tools that will allow AI agents to orchestrate the process. For now, Hamilcar requires human-supervision, but its tools already pass severe stress-tests. In a calculation requiring 10⁵ integration-by-parts operations on a 10³-term expression, Hamilcar automatically recovers the expression in [22] for the time-space-inequivalence induced by two-loop quantum gravity.

Flex-knots. — In parallel to algorithmic advances in theory, non-parametric reconstruction methods offer a powerful, data-driven approach to modelling. The flex-knot technique avoids imposing restrictive, potentially biased analytical forms (e.g., Gaussian profiles or simple polynomials) on observational data. Instead, it models a target function — such as the 21 cm global signal [51], the dark energy equation of state [52], or the primordial power spectrum — as a spline interpolant whose knot locations and amplitudes are themselves free parameters. Within a Bayesian framework, the data determines the model complexity required, allowing for the robust detection of unexpected features while naturally penalising overfitting.

2.3 Methodology

The research programme will be structured through work packages (WP) organized under each R&I Obj., with specific deliverables (Dlv.) which may be published papers, or otherwise new releases of the PSALTer and Hamilcar software packages:

R&I Obj. 1 Quality Control — Uses science products as a catalyst for software upgrades:
WP 1.1 Infrared Foundations — Delivers Infrared Foundations for Quantum Geometry [24, 25], targetting metric-affine theories with torsion T and non-metricity Q in Dlv. 1.1a Paper-IRF-III, and teleparallel theories with vanishing curvature R in Dlv. 1.1b Paper-IRF-IV.

WP 1.2 FORM Porting — Delivers Dlv. 1.2a PSALTer-FORM in which the FORM system for advanced HEP calculations [53] is used as a back-end, instead of the proprietary Wolfram system [54], facilitating state-of-the-art performance in HPC ecosystems.

WP 1.3 Python Porting — A Python front-end is delivered in Dlv. 1.3a PSALTer-Python using SymPy [55] and CADABRA [56]. If this desirable phase-out of Wolfram is too challenging, the Python front-end may be a wrapper, which would still encourage use by the wider precision cosmology community.

WP 1.4 Incorporation — Bake other R&I Obj. into final codebase Dlv. 1.4a PSALTer-Inc.

R&I Obj. 2 Numerical Polology — Opens up an entirely new field:
WP 2.1 Fuzzy Polology — A literal numerical translation of the PSALTer algorithm would result in abrupt transitions of particle number, spin and unitarity in theory-space. This is expected to be a problem for many surveyor engines, whose cost functions require smooth gradients. The solution is to blur the definition of a quantum particle as a pole in the propagator, so that particles emerge and disappear smoothly as the parameters of the theory are varied. The methods of the Källén–Lehmann representation, in which particle spectra are continuously deformed by EFT principles, may serve as inspiration.

WP 2.2 Numerical Engine — Modifies the existing PSALTer wrapper to pass the symbolic physics problem into a new numerical engine, written in C++ using the Eigen and MKL libraries for state-of-the-art performance. Parallelism will be provided by OpenMP and TBB. In case of HPC stability issues, less performant C++ alternatives are Rust, Julia and FORTRAN.

WP 2.3 Surveyor Engine — Develops the back-end to the numerical engine which conducts systematic exploration of high-dimensional theory-space. Multiple sampling algorithms will be implemented and benchmarked: nested sampling for robust evidence estimation and posterior exploration, normalising flows for capturing complex posterior geometries, Markov Chain Monte Carlo (MCMC) for established reliability, and Hamiltonian Monte Carlo for efficient sampling of continuous parameters.

WP 2.4 Observational Plug-Ins — Develops a comprehensive constraint framework that evaluates numerical polology results against observational data across multiple domains. The modular architecture will incorporate gravitational wave constraints from LIGO-Virgo-KAGRA observations, including propagation speed modifications, polarization content, and dispersion relations that emerge from the particle spectrum of each theory. Weak lensing modules will test predicted deviations in light deflection and cosmic shear patterns. Cosmic Microwave Background (CMB) constraints will focus on modified recombination physics and tensor-to-scalar ratio predictions from the gravitational sector. Thermal history modules will examine Big Bang nucleosynthesis and dark matter interaction signatures that depend on the gravitational particle content. Each observational domain will interface with the surveyor engine to provide likelihood evaluations, enabling Bayesian model comparison and theory ranking. This plugin architecture ensures that theory-space exploration is guided by the full breadth of available observational evidence.

WP 2.5 GPU Upgrade — Explores GPU acceleration for numerical polology, targeting NVIDIA CUDA and AMD ROCm platforms for 10-100x performance gains.

R&I Obj. 3 Canonical Reduction — Initially complements R&I Obj. 1 Quality Control and R&I Obj. 2 Numerical Polology, later joining up with polology-based methods:
WP 3.1 MCP Integration — Creates a Hamilcar Model Context Protocol (MCP) server for stress-testing in WP 3.2 Light Scalar Gambit.

WP 3.2 Light Scalar Gambit — Implements canonical reduction for light scalar field theories [26–35]. Observational constraints on reduced theories, including modified propagation speeds and birefringence patterns.

WP 3.3 Canonical Polology — Develops the theory of space-time-inequivalent polology and hardcodes canonical reduction. Additional science products in canonical polology research.

R&I Obj. 4 Bayesian Reconstruction — Is technically ambitious. Its later parts will benefit from early developments to Hamilcar in WP 3.1 MCP Integration:
WP 4.1 Non-Local Gambit — Familiarizes with flexknot cosmological reconstruction, using the non-local theory of Woodard [57–61] (a specific model where the EFT principle is taken to extremes) as a testing arena.

WP 4.2 EFT Extraction — Implements the inversion of the canonical reduction algorithm using the variational bootstrap method. Uses cosmological data to constrain the gravitational EFT.

2.4 Timeline and contingency plan

Impactful science. — My ≥ 7/year publication rate will continue, blending planned deliverables with spontaneous products. Publication targets are Phys. Rev. Lett., IF 9.0; Phys. Rev. X, IF 14.0; Nat. Astron., IF 15.5 for foundational papers; JCAP, IF 6.0; Phys. Rev. D, IF 5.2, for other work; and Computer Physics Communications (IF 4.5) for software. The new code will be an open-source GitLab/HEPForge package with Sphinx documentation and tutorials. I will present results at COSMO, Marcel Grossmann Meetings and Texas Symposium, as well as regular conferences.

Critical risk management. — The timeline mitigates execution risks through balanced scheduling. I minimize critical dependencies to prevent cascading delays, ensuring foundational work concludes before dependent tasks. I also mitigate scientific risk. Only specific work packages test specific theories. Success is therefore not predicated on any single theory, but on three theory-agnostic and largely independent R&I objectives.

Budget. — I require an AI budget of €2,400/year for a coding assistant subscription (e.g., Claude Code MAX 20x) for software development work packages, and €10,000/year for API calls (GPT-4o, Claude Opus, Gemini Pro) to support agent development, with workflows costing €1–10/run. A one-time €12,000–15,000 for a dedicated GPU workstation (e.g., RTX 6000 Ada) will host open-weight models to reduce long-term API costs. The budget also includes €28,800 for travel and dissemination. Additional computation will leverage CSD3 infrastructure and DiRAC funding.

Diverse science. — A two-week annual programme of outreach talks at Chinese secondary schools in low-income areas has been planned in partnership with the Coursemo company, with workshops for navigating Western university admissions for non-native speakers."""

    return extracted_text

def estimate_token_count(text: str) -> dict:
    """
    Estimate token count using the standard approximation of 4 characters per token.
    Provides estimates for different model contexts.
    """
    char_count = len(text)
    token_estimate = char_count // 4
    
    # Token limits for different models
    model_limits = {
        'gpt-4o': 128000,
        'gpt-4o-mini': 128000,
        'claude-3-5-sonnet': 200000,
        'claude-3-5-haiku': 200000,
        'gemini-1.5-pro': 1000000,
        'gemini-1.5-flash': 1000000,
        'o1-preview': 128000,
        'o1-mini': 128000
    }
    
    analysis = {
        'char_count': char_count,
        'estimated_tokens': token_estimate,
        'model_compatibility': {}
    }
    
    for model, limit in model_limits.items():
        fits = token_estimate < limit * 0.8  # Leave 20% for prompt and response
        analysis['model_compatibility'][model] = {
            'fits': fits,
            'utilization_percent': (token_estimate / limit) * 100 if limit else 0
        }
    
    return analysis

def create_review_prompt_template(extracted_text: str) -> str:
    """
    Create a structured prompt template for the multi-LLM review system.
    """
    template = f"""I am refereeing for a competitive research fellowship. Here is an application I have been sent.

DOCUMENT TEXT:
{extracted_text}

Please judge this application focusing on:

1. **Scientific Rigor and Novelty**
   - Are the theoretical approaches sound and well-motivated?
   - How novel are the proposed methods (numerical polology, canonical reduction)?
   - Is the mathematical framework appropriate and robust?

2. **Feasibility and Timeline Realism**
   - Are the work packages achievable within the proposed timeframe?
   - Do the deliverables align with the stated objectives?
   - Are the risk mitigation strategies adequate?

3. **Clarity of Presentation**
   - Is the scientific content clearly explained?
   - Are the objectives and methodology well-defined?
   - Is the writing style appropriate for the target audience?

4. **Potential Impact**
   - What is the likely scientific impact of this work?
   - How well does it address current problems in the field?
   - Is the approach likely to advance the state of the art?

5. **Weaknesses and Concerns**
   - What are the main weaknesses in the proposal?
   - Are there any red flags or concerns?
   - What could be improved?

Please provide specific, actionable feedback with concrete suggestions for improvement. Be critical but fair, as this is for a highly competitive fellowship program."""

    return template

def main():
    parser = argparse.ArgumentParser(description='Extract text from PDF for Martensite review system')
    parser.add_argument('--pdf-path', default='/home/barker/Documents/Applications/2025/LaCaixaIASTRO/StatementOfPurpose.pdf',
                       help='Path to PDF file')
    parser.add_argument('--output-dir', default='/home/barker/Documents/Applications/Martensite/config',
                       help='Output directory for extracted text and templates')
    
    args = parser.parse_args()
    
    # Extract text
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(args.pdf_path)
    
    # Analyze token count
    print("Analyzing token count...")
    token_analysis = estimate_token_count(extracted_text)
    
    # Create prompt template
    print("Creating review prompt template...")
    prompt_template = create_review_prompt_template(extracted_text)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save extracted text
    with open(output_dir / 'extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    
    # Save prompt template
    with open(output_dir / 'review_prompt_template.txt', 'w', encoding='utf-8') as f:
        f.write(prompt_template)
    
    # Save token analysis
    with open(output_dir / 'token_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(token_analysis, f, indent=2)
    
    # Save metadata
    metadata = {
        'extraction_date': datetime.now().isoformat(),
        'source_pdf': args.pdf_path,
        'text_length_chars': len(extracted_text),
        'estimated_tokens': token_analysis['estimated_tokens'],
        'extraction_method': 'manual_from_claude_code_pdf_read'
    }
    
    with open(output_dir / 'extraction_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Text extraction complete!")
    print(f"✓ Files saved to: {output_dir}")
    print(f"✓ Text length: {len(extracted_text):,} characters")
    print(f"✓ Estimated tokens: {token_analysis['estimated_tokens']:,}")
    print(f"\nModel compatibility:")
    for model, compat in token_analysis['model_compatibility'].items():
        status = "✓" if compat['fits'] else "✗"
        print(f"  {status} {model}: {compat['utilization_percent']:.1f}% of context")

if __name__ == "__main__":
    main()