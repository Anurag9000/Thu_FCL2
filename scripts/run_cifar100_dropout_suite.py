#!/usr/bin/env python3
"""Run the CIFAR-100 suite with one random client absent in both sync directions."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cifar100_suite import main

sys.argv.append("--participation-mode")
sys.argv.append("drop_one")
raise SystemExit(main())
