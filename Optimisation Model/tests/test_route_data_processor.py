#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:23:36 2026

@author: mollyrobertson
"""

import os
import pandas as pd
import numpy as np
from route_data_processor import RouteDataProcessor

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))