"""Steps 3 & 4 — simulator + bootstrap classifier."""

from uil.classifier.rule_classifier import RuleClassifier
from uil.domain.labels import ImpairmentLabel
from uil.sim.spectrum_simulator import SpectrumSimulator


def test_clean_spectrum_classified_clean() -> None:
    sim = SpectrumSimulator(seed=1)
    clf = RuleClassifier()
    spec = sim.generate(ImpairmentLabel.Clean)
    c = clf.classify_spectrum(spec, device_type="RPD", measurement_id="m1")
    assert c.status == "clean"
    assert c.klass == ImpairmentLabel.Clean


def test_cpd_raises_floor_and_is_detected() -> None:
    sim = SpectrumSimulator(seed=1)
    clf = RuleClassifier()
    clean = sim.generate(ImpairmentLabel.Clean)
    cpd = sim.generate(ImpairmentLabel.CPD)

    import numpy as np
    low = lambda s: float(np.median([v for i, v in enumerate(s.traces.average) if s.bin_center_hz(i) <= 42e6]))
    assert low(cpd) > low(clean) + 4.0  # floor lifted by the configured margin

    c = clf.classify_spectrum(cpd, device_type="AMP", measurement_id="m2", amp_id="A2")
    assert c.status == "impaired"
    assert c.klass == ImpairmentLabel.CPD
    assert c.confidence >= 0.7


def test_spectrum_trace_lengths_match_numbins() -> None:
    sim = SpectrumSimulator()
    spec = sim.generate(ImpairmentLabel.CPD)
    assert len(spec.traces.average) == spec.numBins == 256
    assert len(spec.traces.maxHold) == len(spec.traces.minHold) == 256
