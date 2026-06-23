"""
Tests for the IP_ADDRESS and HOSTNAME categories (server-topology protection).

Masks internal/public IPv4 and internal-TLD hostnames so server network
identifiers don't leak to an external LLM during log analysis — while leaving
public domains, version numbers' look-alikes minimal, and ports untouched.
"""
from __future__ import annotations

import pytest

from pii_guard.engine import Engine


@pytest.fixture(scope="module")
def engine():
    return Engine()  # Stage-1 only — IP/HOSTNAME are regex categories


def _cats(engine, text):
    return {(d.category, d.original) for d in engine.scan(text).detections}


# ── IP_ADDRESS — detected ─────────────────────────────────────────────────────
@pytest.mark.parametrize("ip", [
    "10.0.12.45", "192.168.1.100", "172.16.5.9", "203.0.113.55", "255.255.255.0",
])
def test_ipv4_detected(engine, ip):
    assert ("IP_ADDRESS", ip) in _cats(engine, f"host {ip} down")


def test_ip_inside_url(engine):
    cats = _cats(engine, "db postgres://u@10.0.0.5:5432/x")
    assert any(c == "IP_ADDRESS" and v == "10.0.0.5" for c, v in cats)


# ── IP_ADDRESS — NOT over-matched ─────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "날짜 2026.06.23 입니다",   # 3 parts
    "포트 8080 사용",
    "octet 256.1.1.1 invalid",  # 256 out of range
    "버전 v1.2.3 빌드",
])
def test_ip_not_false_positive(engine, text):
    assert not any(c == "IP_ADDRESS" for c, _ in _cats(engine, text))


# ── HOSTNAME — internal FQDN detected, public NOT ─────────────────────────────
@pytest.mark.parametrize("host", [
    "prod-payment-svc-02.internal", "db-01.corp", "cache.local",
    "auth.svc.lan", "gw.intranet",
])
def test_internal_hostname_detected(engine, host):
    assert ("HOSTNAME", host) in _cats(engine, f"connect to {host} failed")


@pytest.mark.parametrize("host", [
    "api.anthropic.com", "gmail.com", "www.naver.com", "github.com",
])
def test_public_domain_not_masked(engine, host):
    assert not any(c == "HOSTNAME" for c, _ in _cats(engine, f"calling {host} ok"))


def test_ip_and_hostname_both_masked_and_action(engine):
    r = engine.scan("ERROR peer=10.0.5.5 host=prod-api-01.internal timeout")
    cats = {d.category for d in r.detections}
    assert {"IP_ADDRESS", "HOSTNAME"} <= cats
    # server topology → mask (not block) so structure is preserved for analysis
    assert r.has_masks and not r.has_blocks
