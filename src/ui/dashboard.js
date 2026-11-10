document.addEventListener("DOMContentLoaded", () => {
  // -------------------------------------------------------------
  // ThreeUI PredictiveArcCanvas Integration (SHA-256 fa86582fc870)
  // Variant: Predictive Arc (predictive)
  // Configured usage: mode="dark", speed=1.00, hue=0, saturation=1.00, brightness=1.00
  // -------------------------------------------------------------
  function initPredictiveArcBackground() {
    const host = document.getElementById("predictiveArcHost");
    const canvas = document.getElementById("predictiveArcCanvas");
    if (!host || !canvas) return;

    const PREDICTIVE_ARC_OPTIONS = {
      mode: "dark",
      speed: 1.0,
      spacing: 5,
      dotSize: 6,
      archHeight: 0.7,
      thickness: 1,
      brightness: 1.0,
      hue: 0,
      saturation: 1.0,
    };

    function resolveMode(mode) {
      if (mode === "light" || mode === 1 || mode === "1") return "light";
      return "dark";
    }

    function createPredictiveArcRenderer(canvasEl, getOptions) {
      const context = canvasEl.getContext("2d", { alpha: false });
      if (!context) return null;
      let width = 1;
      let height = 1;
      let time = 0;

      const resize = (nextWidth, nextHeight) => {
        width = Math.max(1, nextWidth);
        height = Math.max(1, nextHeight);
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        canvasEl.width = Math.round(width * pixelRatio);
        canvasEl.height = Math.round(height * pixelRatio);
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      };

      const render = () => {
        const options = getOptions();
        const mode = resolveMode(options.mode);
        const isLight = mode === "light";
        context.fillStyle = isLight ? "#eef1f6" : "#030303";
        context.fillRect(0, 0, width, height);
        time += 0.015 * options.speed;

        const centerX = width / 2;
        const archPeakY = height * 0.35;
        const archWidth = width * 1.5;
        const archHeight = height * options.archHeight;
        context.globalCompositeOperation = isLight ? "source-over" : "lighter";

        for (let x = 0; x < width; x += options.spacing) {
          const normX = (x - centerX) / (archWidth / 2);
          const curveY = archPeakY + normX * normX * archHeight;
          for (let y = 0; y < height; y += options.spacing) {
            const distanceToCurve = Math.abs(y - curveY);
            const thickness = (140 + (1 - Math.abs(normX)) * 80) * options.thickness;
            if (distanceToCurve >= thickness) continue;
            let intensity = 1 - distanceToCurve / thickness;
            const waveX = Math.sin(x * 0.015 + time);
            const waveY = Math.cos(y * 0.02 + time);
            intensity = intensity * 0.7 + waveX * waveY * 0.3 * intensity;
            intensity *= Math.max(0, 1 - Math.pow(Math.abs(normX), 2.5));
            if (intensity <= 0.02) continue;

            let r, g, b;
            if (isLight) {
              r = Math.min(255, 48 * intensity + 70 * Math.pow(intensity, 3));
              g = Math.min(255, 28 * intensity + 45 * Math.pow(intensity, 4));
              b = Math.min(255, 120 * intensity + 110 * Math.pow(intensity, 2));
              if (intensity > 0.7) {
                const coreBoost = (intensity - 0.7) * 3.3;
                r = Math.min(255, r + 90 * coreBoost);
                g = Math.min(255, g + 70 * coreBoost);
                b = Math.min(255, b + 110 * coreBoost);
              }
            } else {
              r = Math.min(255, 60 * intensity + 100 * Math.pow(intensity, 3));
              g = Math.min(255, 20 * intensity + 60 * Math.pow(intensity, 4));
              b = Math.min(255, 120 * intensity + 135 * Math.pow(intensity, 2));
              if (intensity > 0.7) {
                const coreBoost = (intensity - 0.7) * 3.3;
                r = Math.min(255, r + 150 * coreBoost);
                g = Math.min(255, g + 150 * coreBoost);
                b = Math.min(255, b + 150 * coreBoost);
              }
            }
            context.fillStyle = `rgb(${Math.floor(r * options.brightness)}, ${Math.floor(g * options.brightness)}, ${Math.floor(b * options.brightness)})`;
            context.fillRect(x, y, options.dotSize * intensity, options.dotSize * intensity);
          }
        }
        context.globalCompositeOperation = "source-over";
      };

      return { resize, render };
    }

    const renderer = createPredictiveArcRenderer(canvas, () => PREDICTIVE_ARC_OPTIONS);
    if (!renderer) return;

    let frame = 0;
    let visible = true;

    const resize = () => {
      const bounds = host.getBoundingClientRect();
      renderer.resize(bounds.width, bounds.height);
      renderer.render();
    };

    const tick = () => {
      renderer.render();
      frame = visible && !document.hidden ? requestAnimationFrame(tick) : 0;
    };

    const observer = new ResizeObserver(resize);
    const intersection = new IntersectionObserver(([entry]) => {
      visible = entry?.isIntersecting ?? true;
      if (visible && !frame) frame = requestAnimationFrame(tick);
      if (!visible && frame) {
        cancelAnimationFrame(frame);
        frame = 0;
      }
    });

    const visibility = () => {
      if (document.hidden && frame) {
        cancelAnimationFrame(frame);
        frame = 0;
      } else if (!document.hidden && visible && !frame) {
        frame = requestAnimationFrame(tick);
      }
    };

    observer.observe(host);
    intersection.observe(host);
    document.addEventListener("visibilitychange", visibility);
    window.addEventListener("resize", resize);

    resize();
    frame = requestAnimationFrame(tick);
  }

  // Launch PredictiveArcCanvas
  initPredictiveArcBackground();

  // Header Elements
  const healthDot = document.getElementById("healthDot");
  const healthText = document.getElementById("healthText");
  const deviceBadge = document.getElementById("deviceBadge");

  // Input & Control Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const videoPreview = document.getElementById("videoPreview");
  const previewContainer = document.getElementById("previewContainer");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdVal = document.getElementById("thresholdVal");
  const resetThresholdBtn = document.getElementById("resetThresholdBtn");
  const engineSelect = document.getElementById("engineSelect");
  const engineBadge = document.getElementById("engineBadge");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const compareBtn = document.getElementById("compareBtn");

  // Meta & Container Elements
  const singleEngineContainer = document.getElementById("singleEngineContainer");
  const comparisonContainer = document.getElementById("comparisonContainer");
  const forensicAuditContainer = document.getElementById("forensicAuditContainer");
  const compConsensusBanner = document.getElementById("compConsensusBanner");
  const compConsensusIcon = document.getElementById("compConsensusIcon");
  const compConsensusTitle = document.getElementById("compConsensusTitle");
  const compConsensusSubtitle = document.getElementById("compConsensusSubtitle");
  const compDiscordanceSummary = document.getElementById("compDiscordanceSummary");

  // Forensic Audit & Evidence Elements
  const forensicEvidenceBox = document.getElementById("forensicEvidenceBox");
  const evidenceStrengthBadge = document.getElementById("evidenceStrengthBadge");
  const forensicEvidenceText = document.getElementById("forensicEvidenceText");
  const tabAiReport = document.getElementById("tabAiReport");
  const tabRawTelemetry = document.getElementById("tabRawTelemetry");
  const refreshAiBtn = document.getElementById("refreshAiBtn");
  const aiAnalystBox = document.getElementById("aiAnalystBox");
  const aiAnalystText = document.getElementById("aiAnalystText");
  const forensicVerdictBadge = document.getElementById("forensicVerdictBadge");
  const forensicVerdictIcon = document.getElementById("forensicVerdictIcon");
  const forensicVerdictText = document.getElementById("forensicVerdictText");
  const forensicCiPill = document.getElementById("forensicCiPill");
  const forensicProbVal = document.getElementById("forensicProbVal");
  const forensicConfidenceLevel = document.getElementById("forensicConfidenceLevel");
  const forensicKinEnergy = document.getElementById("forensicKinEnergy");
  const forensicTempStability = document.getElementById("forensicTempStability");
  const forensicWindowsCount = document.getElementById("forensicWindowsCount");
  const forensicTotalFrames = document.getElementById("forensicTotalFrames");
  const forensicSaliencyImg = document.getElementById("forensicSaliencyImg");
  const forensicSaliencyPlaceholder = document.getElementById("forensicSaliencyPlaceholder");
  const forensicWarningsBox = document.getElementById("forensicWarningsBox");
  const forensicWarningsList = document.getElementById("forensicWarningsList");
  const forensicWindowsList = document.getElementById("forensicWindowsList");

  // Comparison Engine Cards
  const cardVerdictM13 = document.getElementById("cardVerdictM13");
  const cardFillM13 = document.getElementById("cardFillM13");
  const cardFreqM13 = document.getElementById("cardFreqM13");
  const cardTextureM13 = document.getElementById("cardTextureM13");

  const cardVerdictM18 = document.getElementById("cardVerdictM18");
  const cardFillM18 = document.getElementById("cardFillM18");
  const cardKinM18 = document.getElementById("cardKinM18");
  const cardGateM18 = document.getElementById("cardGateM18");

  const cardVerdictM22 = document.getElementById("cardVerdictM22");
  const cardFillM22 = document.getElementById("cardFillM22");
  const cardLogitM22 = document.getElementById("cardLogitM22");

  // Meta & Reset Elements
  const videoMetaBox = document.getElementById("videoMetaBox");
  const targetFileName = document.getElementById("targetFileName");
  const activeEngineLabel = document.getElementById("activeEngineLabel");
  const resetBtn = document.getElementById("resetBtn");

  // Rejection & Diagnostic Elements
  const rejectAlertBanner = document.getElementById("rejectAlertBanner");
  const rejectAlertTitle = document.getElementById("rejectAlertTitle");
  const rejectAlertDesc = document.getElementById("rejectAlertDesc");
  const diagnosticContainer = document.getElementById("diagnosticContainer");
  const diagOodBadge = document.getElementById("diagOodBadge");
  const diagUncBadge = document.getElementById("diagUncBadge");
  const diagQualityBadge = document.getElementById("diagQualityBadge");
  const techDetailsToggle = document.getElementById("techDetailsToggle");
  const techDetailsContent = document.getElementById("techDetailsContent");
  const techConfidenceVal = document.getElementById("techConfidenceVal");
  const techProbVal = document.getElementById("techProbVal");
  const techVarVal = document.getElementById("techVarVal");
  const techEntropyVal = document.getElementById("techEntropyVal");
  const techOodVal = document.getElementById("techOodVal");
  const techFacesVal = document.getElementById("techFacesVal");
  const techBranchContainer = document.getElementById("techBranchContainer");
  const techBranchFused = document.getElementById("techBranchFused");
  const techBranchComp = document.getElementById("techBranchComp");
  const techBranchFreq = document.getElementById("techBranchFreq");
  const techBranchTexture = document.getElementById("techBranchTexture");
  const techBranchAux = document.getElementById("techBranchAux");
  const techBranchKinRow = document.getElementById("techBranchKinRow");
  const techBranchKinematic = document.getElementById("techBranchKinematic");
  const techGateRow = document.getElementById("techGateRow");
  const techGateVal = document.getElementById("techGateVal");

  // Verdict & Metric Elements
  const verdictBox = document.getElementById("verdictBox");
  const verdictText = document.getElementById("verdictText");
  const verdictConfidence = document.getElementById("verdictConfidence");
  const decisionReason = document.getElementById("decisionReason");
  const decisionExplanation = document.getElementById("decisionExplanation");
  const scoreVal = document.getElementById("scoreVal");
  const scoreFill = document.getElementById("scoreFill");
  const rawLogitVal = document.getElementById("rawLogitVal");
  const framesVal = document.getElementById("framesVal");
  const facesVal = document.getElementById("facesVal");

  // Latency Elements
  const prepLatencyVal = document.getElementById("prepLatencyVal");
  const modelLatencyVal = document.getElementById("modelLatencyVal");
  const totalLatencyVal = document.getElementById("totalLatencyVal");
  const prepLatencyBar = document.getElementById("prepLatencyBar");
  const modelLatencyBar = document.getElementById("modelLatencyBar");

  // Provenance Elements
  const provEngine = document.getElementById("provEngine");
  const provArch = document.getElementById("provArch");
  const provBackbone = document.getElementById("provBackbone");
  const provCalibration = document.getElementById("provCalibration");
  const provDataset = document.getElementById("provDataset");
  const provPaper = document.getElementById("provPaper");

  let selectedFile = null;
  let lastPredictionData = null;
  let lastAuditData = null;

  // 1. Fetch Health Status
  fetch("/health")
    .then((res) => res.json())
    .then((data) => {
      if (data.status === "healthy") {
        healthDot.classList.add("online");
        healthText.textContent = "System Operational";
        if (data.device) {
          deviceBadge.textContent = data.device.toUpperCase() + (data.device.includes("cuda") ? " Acceleration" : " Mode");
        }
      } else {
        healthText.textContent = "Service Ready (Unloaded)";
      }
    })
    .catch(() => {
      healthText.textContent = "Backend Offline";
    });

  // 2. Fetch Model Info & Provenance
  function loadModelInfo(engine = "v5-family-invariant") {
    fetch(`/api/v1/model-info?engine=${encodeURIComponent(engine)}`)
      .then((res) => res.json())
      .then((info) => {
        if (provEngine) provEngine.textContent = info.display_name || info.engine;
        if (provArch) provArch.textContent = info.architecture || "EfficientNet-B0 + Temporal Transformer";
        if (provBackbone) provBackbone.textContent = info.backbone || "efficientnet_b0";
        if (provCalibration) {
          provCalibration.textContent = info.calibration ? `${info.calibration.dataset} (${info.calibration.version})` : "Standard Binary Baseline";
        }
        if (provDataset) provDataset.textContent = info.training_dataset || "FaceForensics++ C23";
        if (provPaper) provPaper.textContent = info.reference_paper || "Lal et al., 2025 / M12 Robust";
        if (info.active_device && deviceBadge) {
          deviceBadge.textContent = info.active_device.toUpperCase() + (info.active_device.includes("cuda") ? " Acceleration" : " Mode");
        }
      })
      .catch((err) => {
        console.warn("Could not load model info:", err);
      });
  }

  // Load default engine info (v5-family-invariant)
  const initialEngine = engineSelect ? engineSelect.value : "v5-family-invariant";
  loadModelInfo(initialEngine);

  // Engine Switch Listener
  if (engineSelect) {
    engineSelect.addEventListener("change", (e) => {
      const selected = e.target.value;
      const engineTitles = {
        "v5-family-invariant": "v5-family-invariant (Active Production Champion)",
        "m22-ensemble": "m22-ensemble (Highest-Accuracy Decision Fusion Ensemble)",
        "m18-dual-stream": "m18-dual-stream (Reenactment Baseline — 3D Kinematics)",
        "m17-sensor-residuals": "m17-sensor-residuals (Sensor Noise Baseline)",
        "v4-generalization": "v4-generalization (M13.2b)",
        "v3-robust": "v3-robust (M12)",
        "v2-evolved": "v2-evolved (M10)",
        "v1-baseline": "v1-baseline",
      };
      if (engineBadge) {
        engineBadge.textContent = engineTitles[selected] || selected;
      }
      // Auto-set operating threshold for calibrated models
      if (selected === "m22-ensemble") {
        thresholdSlider.value = "0.92";
        thresholdVal.textContent = "0.92";
      } else {
        thresholdSlider.value = "0.50";
        thresholdVal.textContent = "0.50";
      }
      loadModelInfo(selected);
    });
  }

  // Toggle Technical Uncertainty Metrics
  if (techDetailsToggle && techDetailsContent) {
    techDetailsToggle.addEventListener("click", () => {
      const isHidden = techDetailsContent.style.display === "none";
      techDetailsContent.style.display = isHidden ? "block" : "none";
      techDetailsToggle.textContent = isHidden ? "Hide Technical Metrics ▲" : "Show Technical Metrics ▼";
    });
  }

  // Safe numeric formatting helpers
  function formatNumber(val, decimals = 2, fallback = "--") {
    if (val === undefined || val === null || isNaN(Number(val))) {
      return fallback;
    }
    return Number(val).toFixed(decimals);
  }

  function formatScore(val, decimals = 4, suffix = "") {
    if (val === undefined || val === null || isNaN(Number(val))) {
      return "--";
    }
    return `p = ${Number(val).toFixed(decimals)}${suffix}`;
  }

  // 3. Dynamic Threshold & Verdict Rendering (3-State Calibrated)
  function applyThresholdAndRenderVerdict(data, threshold) {
    if (!data) return;

    const isReject = data.decision === "REJECT" || data.verdict === "Reject";

    if (isReject) {
      verdictBox.className = "verdict-box reject";
      verdictText.textContent = "DECISION: REJECT";
      verdictConfidence.textContent = "Evidence Insufficient (0.0%)";

      // Show explicit rejection banner (REJECT ≠ FAKE)
      if (rejectAlertBanner) {
        rejectAlertBanner.style.display = "flex";
        if (rejectAlertTitle) {
          rejectAlertTitle.textContent = `OPERATING ENVELOPE REJECTION [${data.rejection_code || "REJECT"}]`;
        }
        if (rejectAlertDesc) {
          const detail = data.rejection_reason || "Insufficient facial evidence within validated envelope.";
          rejectAlertDesc.innerHTML = `<strong>REJECT ≠ FAKE.</strong> ${detail} The model declines to make a high-stakes binary attribution because error risk exceeds safety constraints.`;
        }
      }

      if (decisionReason) decisionReason.style.display = "none";
    } else {
      // Hide rejection banner for valid predictions
      if (rejectAlertBanner) rejectAlertBanner.style.display = "none";

      const prob = Number(data.probability ?? 0);
      const isFake = prob >= threshold;

      verdictBox.className = `verdict-box ${isFake ? "deepfake" : "real"}`;
      verdictText.textContent = isFake ? "PREDICTION: DEEPFAKE" : "PREDICTION: REAL";

      const pFake = prob;
      const pctFake = (pFake * 100).toFixed(2);
      verdictConfidence.textContent = `Model Score: ${pFake.toFixed(4)} (P(Fake): ${pctFake}%)`;

      if (decisionReason && decisionExplanation) {
        decisionReason.style.display = "block";
        const probPct = (prob * 100).toFixed(1);
        const threshPct = (threshold * 100).toFixed(1);

        if (isFake) {
          decisionExplanation.innerHTML = `<span style="color: var(--state-error);">● <strong>Deepfake Detected:</strong> Calibrated probability <strong>${probPct}%</strong> ≥ Threshold <strong>${threshPct}%</strong>.</span>`;
          decisionReason.style.borderColor = "var(--state-error)";
          decisionReason.style.background = "rgba(240, 112, 112, 0.08)";
        } else {
          if (prob >= 0.50) {
            decisionExplanation.innerHTML = `<span style="color: var(--state-warning);">⚠ <strong>Classified as Real due to high threshold (${threshPct}%).</strong> The model observed deepfake features (${probPct}%), but threshold requires ≥ ${threshPct}%.</span>`;
            decisionReason.style.borderColor = "var(--state-warning)";
            decisionReason.style.background = "rgba(231, 183, 91, 0.08)";
          } else {
            decisionExplanation.innerHTML = `<span style="color: var(--state-success);">● <strong>Authentic Video:</strong> Deepfake score <strong>${probPct}%</strong> &lt; Threshold <strong>${threshPct}%</strong>. Evidence supports authenticity within validated envelope.</span>`;
            decisionReason.style.borderColor = "var(--state-success)";
            decisionReason.style.background = "rgba(98, 197, 139, 0.08)";
          }
        }
      }
    }

    // Diagnostics Container Rendering
    if (diagnosticContainer) {
      diagnosticContainer.style.display = "block";

      const unc = data.uncertainty;
      const qual = data.quality;

      // OOD Status
      if (diagOodBadge) {
        const isOod = unc ? Boolean(unc.is_ood) : false;
        const oodDist = (unc && unc.ood_distance !== undefined && unc.ood_distance !== null) ? Number(unc.ood_distance) : 0.0;
        diagOodBadge.className = `diag-badge ${isOod ? "warn" : "ok"}`;
        diagOodBadge.textContent = isOod ? `Elevated Distance (${oodDist.toFixed(2)})` : `Within Validated Range (${oodDist.toFixed(2)})`;
      }

      // Epistemic Uncertainty Status
      if (diagUncBadge) {
        const isUnc = unc ? Boolean(unc.is_uncertain) : false;
        diagUncBadge.className = `diag-badge ${isUnc ? "warn" : "ok"}`;
        diagUncBadge.textContent = isUnc ? "High Model Variance" : "Low Epistemic Uncertainty";
      }

      // Face Quality Status
      if (diagQualityBadge) {
        const isQualOk = qual ? Boolean(qual.is_acceptable) : true;
        diagQualityBadge.className = `diag-badge ${isQualOk ? "ok" : "error"}`;
        diagQualityBadge.textContent = isQualOk ? "Acceptable Face Sequence" : "Degraded / Missing Face Tracking";
      }

      // Technical Metrics
      if (techConfidenceVal) {
        const probStr = formatNumber(data.probability, 4);
        const logitStr = data.raw_logit !== undefined && data.raw_logit !== null ? formatNumber(data.raw_logit, 3) : "N/A";
        techConfidenceVal.textContent = `${probStr} (Logit: ${logitStr})`;
      }
      if (techProbVal) techProbVal.textContent = formatNumber(data.probability, 4);
      if (techVarVal) techVarVal.textContent = formatNumber(unc?.epistemic_variance, 6);
      if (techEntropyVal) {
        const ent = unc?.entropy ?? unc?.predictive_entropy;
        techEntropyVal.textContent = `${formatNumber(ent, 3)} bits`;
      }
      if (techOodVal) techOodVal.textContent = formatNumber(unc?.ood_distance, 2);
      if (techFacesVal) techFacesVal.textContent = qual ? `${qual.detected_frames ?? "--"} / ${qual.total_frames ?? "--"}` : "--";

      // Multi-Domain Branch Diagnostic Scores (Non-Attributional)
      const branchScores = (data.metadata && data.metadata.branch_scores) || (data.diagnostics && data.diagnostics.branch_scores);
      if (branchScores && techBranchContainer) {
        techBranchContainer.style.display = "block";
        if (techBranchFused) techBranchFused.textContent = formatScore(branchScores.p_fused ?? branchScores.p_primary, 4, " (raw score)");
        if (techBranchComp) techBranchComp.textContent = formatScore(branchScores.p_composite, 4, " (raw score)");
        if (techBranchFreq) techBranchFreq.textContent = formatScore(branchScores.p_freq, 4, " (raw score)");
        if (techBranchTexture) techBranchTexture.textContent = formatScore(branchScores.p_texture, 4, " (raw score)");
        if (techBranchAux) techBranchAux.textContent = formatScore(branchScores.p_aux_fake, 4, " (raw score)");
      } else if (techBranchContainer) {
        techBranchContainer.style.display = "none";
      }
    }
  }

  // Evidence Tab Switching (AI Analyst Report vs Structured Telemetry)
  if (tabAiReport && tabRawTelemetry && aiAnalystBox && forensicEvidenceText) {
    tabAiReport.addEventListener("click", () => {
      tabAiReport.classList.add("active");
      tabRawTelemetry.classList.remove("active");
      aiAnalystBox.style.display = "block";
      forensicEvidenceText.style.display = "none";
    });

    tabRawTelemetry.addEventListener("click", () => {
      tabRawTelemetry.classList.add("active");
      tabAiReport.classList.remove("active");
      aiAnalystBox.style.display = "none";
      forensicEvidenceText.style.display = "block";
    });
  }

  if (refreshAiBtn) {
    refreshAiBtn.addEventListener("click", () => {
      if (!lastAuditData) return;
      const effThreshold = parseFloat(thresholdSlider.value || 0.50);
      fetchGroqSynthesis(lastAuditData, lastPredictionData, effThreshold);
    });
  }

  let thresholdDebounceTimer = null;

  thresholdSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    thresholdVal.textContent = val.toFixed(2);
    if (lastAuditData) {
      renderForensicAudit(lastAuditData, val, lastPredictionData, engineSelect ? engineSelect.value : null);
    }
    if (lastPredictionData) {
      applyThresholdAndRenderVerdict(lastPredictionData, val);
    }

    if (lastAuditData) {
      clearTimeout(thresholdDebounceTimer);
      thresholdDebounceTimer = setTimeout(() => {
        fetchGroqSynthesis(lastAuditData, lastPredictionData, val);
      }, 500);
    }
  });

  if (resetThresholdBtn) {
    resetThresholdBtn.addEventListener("click", () => {
      const isM22 = engineSelect && engineSelect.value === "m22-ensemble";
      const defaultVal = isM22 ? "0.92" : "0.50";
      const defaultNum = parseFloat(defaultVal);
      thresholdSlider.value = defaultVal;
      thresholdVal.textContent = defaultVal;
      if (lastAuditData) {
        renderForensicAudit(lastAuditData, defaultNum, lastPredictionData, engineSelect ? engineSelect.value : null);
        clearTimeout(thresholdDebounceTimer);
        fetchGroqSynthesis(lastAuditData, lastPredictionData, defaultNum);
      }
      if (lastPredictionData) {
        applyThresholdAndRenderVerdict(lastPredictionData, defaultNum);
      }
    });
  }

  // 4. File Drop & Selection
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = `Analyze ${file.name}`;
    if (compareBtn) {
      compareBtn.disabled = false;
      compareBtn.textContent = "⚖️ Compare All Engines";
    }

    // Show local video preview
    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    previewContainer.style.display = "flex";
  }

  // 5a. Single-Engine Prediction Helper (/api/v1/predict)
  function runSingleEnginePrediction(file, engine, threshold) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("threshold", threshold);
    formData.append("engine", engine);
    formData.append("calibrated", "true");
    return fetch("/api/v1/predict", { method: "POST", body: formData })
      .then((res) => {
        if (!res.ok) return res.json().then((err) => { throw new Error(err.detail || "Prediction failed"); });
        return res.json();
      });
  }

  // 5. Submit Video for Selected Engine Analysis & Forensic Evidence
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    const selectedEngine = engineSelect ? engineSelect.value : "v5-family-invariant";
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = `Analyzing with ${selectedEngine}...`;
    if (compareBtn) compareBtn.disabled = true;

    const currentThreshold = parseFloat(thresholdSlider.value);

    try {
      // 1. Fetch prediction from the user-selected engine
      const predFormData = new FormData();
      predFormData.append("file", selectedFile);
      predFormData.append("threshold", currentThreshold);
      predFormData.append("engine", selectedEngine);
      predFormData.append("calibrated", "true");

      const predPromise = fetch("/api/v1/predict", { method: "POST", body: predFormData })
        .then((res) => (res.ok ? res.json() : null))
        .catch(() => null);

      // 2. Fetch forensic audit (saliency heatmap, temporal windows, differential energy)
      const auditFormData = new FormData();
      auditFormData.append("file", selectedFile);
      auditFormData.append("threshold", currentThreshold);
      auditFormData.append("windows", "4");
      auditFormData.append("frames_per_window", "16");

      const auditPromise = fetch("/api/v1/forensic-audit", { method: "POST", body: auditFormData })
        .then((res) => (res.ok ? res.json() : null))
        .catch(() => null);

      const [predData, auditData] = await Promise.all([predPromise, auditPromise]);

      if (auditData) {
        lastAuditData = auditData;
        lastPredictionData = predData;
        renderForensicAudit(auditData, currentThreshold, predData, selectedEngine);
      } else if (predData) {
        lastPredictionData = predData;
        renderPrediction(predData, currentThreshold);
      } else {
        throw new Error("Analysis failed: backend service did not return results");
      }
    } catch (err) {
      alert(`Analysis Error: ${err.message}`);
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = `Analyze ${selectedFile.name}`;
      if (compareBtn) compareBtn.disabled = false;
    }
  });

  // 5b. Submit Video for Multi-Engine Comparison
  if (compareBtn) {
    compareBtn.addEventListener("click", () => {
      if (!selectedFile) return;

      compareBtn.disabled = true;
      compareBtn.textContent = "Comparing Engines (M13.5 + M18 + M22)...";
      if (analyzeBtn) analyzeBtn.disabled = true;

      const currentThreshold = parseFloat(thresholdSlider.value);
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("threshold", currentThreshold);
      formData.append("calibrated", "true");

      fetch("/api/v1/compare", {
        method: "POST",
        body: formData,
      })
        .then((res) => {
          if (!res.ok) {
            return res.json().then((err) => {
              throw new Error(err.detail || "Comparison failed");
            });
          }
          return res.json();
        })
        .then((data) => {
          renderComparison(data);
        })
        .catch((err) => {
          alert(`Comparison Error: ${err.message}`);
        })
        .finally(() => {
          compareBtn.disabled = false;
          compareBtn.textContent = "⚖️ Compare All Engines";
          if (analyzeBtn) analyzeBtn.disabled = false;
        });
    });
  }

  function escapeHtml(str) {
    if (str == null) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatBytes(bytes) {
    if (!bytes || isNaN(bytes)) return "N/A";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let b = Number(bytes);
    while (b >= 1024 && i < units.length - 1) {
      b /= 1024;
      i++;
    }
    return `${b.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
  }

  function renderMarkdownToHtml(md) {
    if (!md) return "";
    let html = escapeHtml(md);

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3 class="ai-analyst-h3">🔍 $1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2 class="ai-analyst-h2">📌 $1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1 class="ai-analyst-h1">🧬 $1</h1>');

    // Bold, Italic, Inline Code
    html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');
    html = html.replace(/`([^`]+)`/gim, '<code>$1</code>');

    // Horizontal Rule
    html = html.replace(/^---$/gim, '<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.6rem 0;">');

    // Bullet Lists
    html = html.replace(/^\s*[\-\*•]\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>(\s*<li>.*<\/li>)*)/gim, '<ul>$1</ul>');

    // Blockquotes & Notes
    html = html.replace(/^>\s*\[!NOTE\]\s*(.*$)/gim, '<div class="ai-threshold-banner" style="border-left-color: #38bdf8; background: rgba(56, 189, 248, 0.08);">$1</div>');
    html = html.replace(/^>\s*(.*$)/gim, '<blockquote style="border-left: 3px solid rgba(255,255,255,0.2); padding-left: 0.8rem; margin: 0.5rem 0; color: #cbd5e1; font-style: italic;">$1</blockquote>');

    // Paragraph wrapping
    const paragraphs = html.split(/\n\s*\n/).map((p) => {
      const trimmed = p.trim();
      if (!trimmed) return "";
      if (
        trimmed.startsWith("<h1") ||
        trimmed.startsWith("<h2") ||
        trimmed.startsWith("<h3") ||
        trimmed.startsWith("<ul") ||
        trimmed.startsWith("<div") ||
        trimmed.startsWith("<hr") ||
        trimmed.startsWith("<blockquote")
      ) {
        return trimmed;
      }
      return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
    });

    html = paragraphs.filter(Boolean).join("");

    return html;
  }

  async function fetchGroqSynthesis(data, predData, threshold) {
    if (!aiAnalystText) return;
    aiAnalystText.innerHTML = `
      <div style="display: flex; align-items: center; gap: 0.6rem; padding: 0.8rem 0; color: #38bdf8; font-size: 0.82rem;">
        <span style="display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(56,189,248,0.25); border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.8s linear infinite;"></span>
        Synthesizing canonical forensic analysis at threshold τ = ${Number(threshold).toFixed(2)}...
      </div>
    `;

    try {
      const resp = await fetch("/api/v1/groq-analyst-synthesis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audit_data: data,
          prediction_data: predData,
          threshold: threshold,
        }),
      });

      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || `HTTP ${resp.status}`);
      }

      const resData = await resp.json();
      if (resData.groq_report) {
        data.groq_analyst_report = resData.groq_report;
        data._last_analyzed_threshold = threshold;
        let validBadge = "";
        if (resData.validation_level === "VERIFIED_CANONICAL") {
          validBadge = `<div style="display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.25rem 0.6rem; background: rgba(52, 211, 153, 0.12); border: 1px solid rgba(52, 211, 153, 0.3); color: #34d399; border-radius: 4px; font-size: 0.68rem; font-weight: 600; margin-bottom: 0.8rem;" title="All numerical claims, decision rules, and causal boundaries verified against current analysis run.">🛡️ Verified against Canonical Evidence</div>`;
        } else if (resData.validation_level === "FIELDS_CHECKED") {
          validBadge = `<div style="display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.25rem 0.6rem; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; border-radius: 4px; font-size: 0.68rem; font-weight: 600; margin-bottom: 0.8rem;" title="Evidence fields checked against canonical pipeline schema.">✓ Evidence fields checked</div>`;
        } else if (resData.validation_level === "FALLBACK_DETERMINISTIC" || resData.is_fallback) {
          validBadge = `<div style="display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.25rem 0.6rem; background: rgba(251, 191, 36, 0.12); border: 1px solid rgba(251, 191, 36, 0.3); color: #fbbf24; border-radius: 4px; font-size: 0.68rem; font-weight: 600; margin-bottom: 0.8rem;" title="Deterministic synthesis compiled directly from canonical measurements.">ℹ️ Deterministic Canonical Report</div>`;
        } else {
          validBadge = `<div style="display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.25rem 0.6rem; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; border-radius: 4px; font-size: 0.68rem; font-weight: 600; margin-bottom: 0.8rem;" title="Validation issues detected during analysis run.">⚠️ Telemetry Alignment Required</div>`;
        }
        aiAnalystText.innerHTML = validBadge + renderMarkdownToHtml(resData.groq_report);
        if (refreshAiBtn) refreshAiBtn.style.display = "inline-block";
      } else {
        throw new Error("No report content received from Groq.");
      }
    } catch (err) {
      aiAnalystText.innerHTML = `
        <div style="color: #f87171; font-size: 0.78rem; padding: 0.5rem 0;">
          ⚠️ Groq AI Synthesis Notice: ${escapeHtml(err.message)}
          <div style="margin-top: 0.5rem;">
            <button type="button" class="tab-chip" onclick="document.getElementById('tabRawTelemetry').click()">View Structured Telemetry</button>
          </div>
        </div>
      `;
    }
  }

  function renderStructuredTelemetry(container, st, fallbackText) {
    if (!container) return;
    if (!st || typeof st !== "object") {
      container.textContent = fallbackText || "No telemetry available.";
      return;
    }

    const stream = st.stream || {};
    const spatial = st.spatial || {};
    const temporal = st.temporal || {};
    const baseline = st.baseline || {};
    const quality = st.quality || {};
    const decision = st.decision || {};
    const peak = temporal.peak_window || {};

    const shaShort = stream.file_sha256 ? `${stream.file_sha256.substring(0, 16)}...` : "Unavailable";
    const shaFull = stream.file_sha256 || "N/A";
    const width = stream.width || stream.frame_width;
    const height = stream.height || stream.frame_height;
    const res = (width && height) ? `${width}×${height}` : "Unavailable";
    const fps = (stream.container_fps != null ? stream.container_fps : stream.fps) ? `${Number(stream.container_fps || stream.fps).toFixed(2)} fps` : "VFR/Unknown";
    const duration = stream.duration_sec != null ? `${Number(stream.duration_sec).toFixed(2)}s` : "Unknown";
    const frameCount = (temporal.total_frames_analyzed != null ? `${temporal.total_frames_analyzed} frames` : (stream.frame_count_decoded != null ? `${stream.frame_count_decoded} frames` : "Unknown"));
    const codec = (stream.codec_name || stream.codec) ? (stream.codec_name || stream.codec).toUpperCase() : "RAW/UNKNOWN";
    const decodeStatus = stream.decode_status || "SUCCESS";
    const fileSize = formatBytes(stream.file_size_bytes);
    const fileName = stream.source_file_name || stream.file_name || "video.mp4";

    const boundaryVarVal = (spatial.outer_boundary_energy != null ? spatial.outer_boundary_energy : spatial.boundary_gradient_variance);
    const boundaryVar = boundaryVarVal != null ? Number(boundaryVarVal).toFixed(2) : "N/A";
    const innerVarVal = (spatial.inner_landmark_energy != null ? spatial.inner_landmark_energy : spatial.inner_gradient_variance);
    const innerVar = innerVarVal != null ? Number(innerVarVal).toFixed(2) : "N/A";
    const ratioVal = (spatial.boundary_to_landmark_ratio != null ? spatial.boundary_to_landmark_ratio : spatial.outer_inner_boundary_ratio);
    const ratio = ratioVal != null ? Number(ratioVal).toFixed(2) : "N/A";
    const spatialAnomaly = Boolean(spatial.spatial_frequency_anomaly_flag || (ratioVal != null && Number(ratioVal) > 1.35));

    const winCount = temporal.total_windows_evaluated != null ? temporal.total_windows_evaluated : (temporal.windows_evaluated != null ? temporal.windows_evaluated : 0);
    const framesTotal = temporal.total_frames_analyzed != null ? temporal.total_frames_analyzed : (temporal.frames_evaluated_total != null ? temporal.frames_evaluated_total : 0);
    const diffRateVal = (temporal.inter_frame_differential_energy != null ? temporal.inter_frame_differential_energy : temporal.frame_differential_rate);
    const diffRate = diffRateVal != null ? Number(diffRateVal).toFixed(2) : "N/A";
    const diffUnit = temporal.differential_energy_unit || "2D Euclidean pixel differential per second";

    const peakCalScore = peak.calibrated_fake_probability != null ? peak.calibrated_fake_probability : peak.calibrated_score;
    const peakScore = peakCalScore != null ? `${(Number(peakCalScore) * 100).toFixed(1)}%` : "N/A";
    const peakStart = peak.start_sec != null ? Number(peak.start_sec).toFixed(1) : "0.0";
    const peakEnd = peak.end_sec != null ? Number(peak.end_sec).toFixed(1) : "0.0";
    const peakConf = peak.face_confidence != null ? `${(Number(peak.face_confidence) * 100).toFixed(0)}%` : "N/A";
    const peakPolicy = peak.selection_method || peak.selection_policy || "Highest calibrated model score across active face tracks";
    const dispVal = temporal.window_score_consistency != null ? temporal.window_score_consistency : temporal.temporal_score_dispersion;
    const dispersion = dispVal != null ? `${(Number(dispVal) * 100).toFixed(1)}%` : "N/A";

    const baselineEligible = Boolean(baseline.is_eligible);
    const refDataset = baseline.reference_dataset || "FaceForensics++ C23";
    const bSampleN = baseline.sample_video_count || baseline.sample_size_n;
    const baselineN = bSampleN ? `(N = ${bSampleN})` : "";
    const kinZ = baseline.kinematic_z_score != null ? `${baseline.kinematic_z_score >= 0 ? '+' : ''}${Number(baseline.kinematic_z_score).toFixed(2)}σ` : "N/A";
    const spRatioZ = baseline.spatial_ratio_z_score != null ? baseline.spatial_ratio_z_score : baseline.boundary_ratio_z_score;
    const ratioZ = spRatioZ != null ? `${spRatioZ >= 0 ? '+' : ''}${Number(spRatioZ).toFixed(2)}σ` : "N/A";

    const coverageRatio = quality.face_coverage_ratio != null ? quality.face_coverage_ratio : quality.mean_face_confidence;
    const coveragePct = coverageRatio != null ? `${(Number(coverageRatio) * 100).toFixed(1)}%` : "N/A";
    const coverageOk = Boolean(quality.sufficient_coverage);
    const confList = quality.confound_codes || quality.confounders_flagged || [];
    const confounders = confList.length > 0 ? confList.join(", ") : "None detected";
    const modFailures = (quality.quality_warnings && quality.quality_warnings.length > 0) ? quality.quality_warnings.join("; ") : (quality.module_failures && quality.module_failures.length > 0 ? quality.module_failures.join(", ") : "None (All modules nominal)");

    const verdict = decision.decision_state || decision.verdict || "INCONCLUSIVE";
    const activeThresh = decision.active_threshold != null ? Number(decision.active_threshold).toFixed(2) : "0.50";
    const probScore = decision.calibrated_probability != null ? decision.calibrated_probability : decision.calibrated_video_score;
    const calScore = probScore != null ? `${(Number(probScore) * 100).toFixed(1)}%` : "N/A";
    const modelVer = `${decision.model_name || 'M20PyramidalDualStream'} (${decision.model_version || '2.0.0'})`;
    const ruleVer = decision.decision_rule_version || "2.0.0";
    const family = decision.attribution || decision.detected_manipulation_family || "Unspecified / Binary Detection Only";
    const limitation = decision.limitations || decision.limitation_statement || "Automated analysis within validated operational domain. Not definitive legal proof.";

    container.innerHTML = `
      <div class="telemetry-container">
        <!-- Section 1: Stream Telemetry -->
        <div class="telemetry-section">
          <div class="telemetry-header">
            <span>📡 1. Stream Telemetry</span>
            <span class="telemetry-badge ${decodeStatus === 'SUCCESS' ? 'badge-ok' : 'badge-warn'}">${decodeStatus}</span>
          </div>
          <div class="telemetry-grid">
            <div class="telemetry-item">
              <span class="telemetry-label">Source File</span>
              <span class="telemetry-val">${escapeHtml(stream.file_name || 'video.mp4')} (${fileSize})</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">SHA-256 Digest</span>
              <span class="telemetry-val" title="${escapeHtml(shaFull)}">${escapeHtml(shaShort)}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Resolution & FPS</span>
              <span class="telemetry-val">${res} • ${fps}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Duration & Frames</span>
              <span class="telemetry-val">${duration} (${frameCount})</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Container Codec</span>
              <span class="telemetry-val">${codec}</span>
            </div>
          </div>
        </div>

        <!-- Section 2: Spatial Observations -->
        <div class="telemetry-section">
          <div class="telemetry-header">
            <span>🔬 2. Spatial Frequency & Boundary Analysis</span>
            <span class="telemetry-badge ${spatialAnomaly ? 'badge-warn' : 'badge-ok'}">${spatialAnomaly ? 'Elevated Frequency' : 'Nominal Distribution'}</span>
          </div>
          <div class="telemetry-grid">
            <div class="telemetry-item">
              <span class="telemetry-label">Spatial Filter</span>
              <span class="telemetry-val">${escapeHtml(spatial.filter_applied || '112×112 Discrete Laplacian')}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Boundary Gradient Var</span>
              <span class="telemetry-val">${boundaryVar}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Inner Landmark Var</span>
              <span class="telemetry-val">${innerVar}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Outer/Inner Ratio</span>
              <span class="telemetry-val">${ratio}</span>
            </div>
          </div>
          <div class="telemetry-desc">${escapeHtml(spatial.physical_interpretation || 'Physical frequency measurement across facial boundary.')}</div>
        </div>

        <!-- Section 3: Temporal Observations & Peak Window -->
        <div class="telemetry-section">
          <div class="telemetry-header">
            <span>⏱️ 3. Temporal Observations & Peak Window</span>
            <span class="telemetry-badge badge-neutral">${winCount} Windows • ${framesTotal} Frames</span>
          </div>
          <div class="telemetry-grid">
            <div class="telemetry-item">
              <span class="telemetry-label">Frame Differential Rate</span>
              <span class="telemetry-val">${diffRate} <span style="font-size: 0.65rem; color: #94a3b8;">(${escapeHtml(diffUnit)})</span></span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Peak Window</span>
              <span class="telemetry-val">Window #${peak.window_id != null ? peak.window_id : 1} (${peakStart}s – ${peakEnd}s)</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Peak Window Score</span>
              <span class="telemetry-val" style="color: ${Number(peak.calibrated_score || 0) >= Number(activeThresh) ? '#f87171' : '#34d399'};">${peakScore} (Face Conf: ${peakConf})</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Window Score Dispersion</span>
              <span class="telemetry-val">${dispersion}</span>
            </div>
          </div>
          <div class="telemetry-desc"><strong style="color: #94a3b8;">Peak Selection Policy:</strong> ${escapeHtml(peak.selection_policy || 'Highest calibrated score across active face tracks')}</div>
        </div>

        <!-- Section 4: Empirical Baseline Comparison -->
        <div class="telemetry-section">
          <div class="telemetry-header">
            <span>📊 4. Empirical Baseline Comparison</span>
            <span class="telemetry-badge ${baselineEligible ? 'badge-ok' : 'badge-neutral'}">${baselineEligible ? 'Eligible & Calibrated' : 'Ineligible / Uncalibrated'}</span>
          </div>
          <div class="telemetry-grid">
            <div class="telemetry-item">
              <span class="telemetry-label">Reference Benchmark</span>
              <span class="telemetry-val">${escapeHtml(refDataset)} ${baselineN}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Kinematic Z-Score</span>
              <span class="telemetry-val">${kinZ}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Boundary Ratio Z-Score</span>
              <span class="telemetry-val">${ratioZ}</span>
            </div>
          </div>
          <div class="telemetry-desc">${escapeHtml(baseline.eligibility_notes || (baselineEligible ? 'Standardized deviations computed against reference population distribution.' : 'Standardized baseline comparison ineligible.'))}</div>
        </div>

        <!-- Section 5: Quality Diagnostics & Confounder Flags -->
        <div class="telemetry-section">
          <div class="telemetry-header">
            <span>⚠️ 5. Quality Diagnostics & Confounder Flags</span>
            <span class="telemetry-badge ${coverageOk ? 'badge-ok' : 'badge-danger'}">${coverageOk ? 'Sufficient Coverage' : 'Insufficient Coverage'}</span>
          </div>
          <div class="telemetry-grid">
            <div class="telemetry-item">
              <span class="telemetry-label">Face Track Coverage</span>
              <span class="telemetry-val">${coveragePct}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Active Confounder Flags</span>
              <span class="telemetry-val">${escapeHtml(confounders)}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Module Failures</span>
              <span class="telemetry-val">${escapeHtml(modFailures)}</span>
            </div>
          </div>
        </div>

        <!-- Section 6: Evidence Assessment & Limitations -->
        <div class="telemetry-section">
          <div class="telemetry-header">
            <span>⚖️ 6. Evidence Assessment & Operational Scope</span>
            <span class="telemetry-badge badge-neutral">${escapeHtml(ruleVer)}</span>
          </div>
          <div class="telemetry-grid">
            <div class="telemetry-item">
              <span class="telemetry-label">Forensic Verdict</span>
              <span class="telemetry-val" style="font-weight: 800; color: ${verdict.includes('MANIPULATION DETECTED') ? '#f87171' : verdict.includes('NO') ? '#34d399' : '#fbbf24'};">${escapeHtml(verdict)}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Score vs Active Threshold</span>
              <span class="telemetry-val">${calScore} <span style="font-size: 0.65rem; color: #94a3b8;">(Thresh: ${activeThresh})</span></span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Manipulation Attribution</span>
              <span class="telemetry-val">${escapeHtml(family)}</span>
            </div>
            <div class="telemetry-item">
              <span class="telemetry-label">Audited Model Version</span>
              <span class="telemetry-val">${escapeHtml(modelVer)}</span>
            </div>
          </div>
          <div class="telemetry-disclaimer">
            <strong>Operational Scope & Limits:</strong> ${escapeHtml(limitation)}
          </div>
        </div>
      </div>
    `;
  }

  function renderForensicAudit(data, threshold = null, predData = null, engineName = null) {
    if (!data) return;

    if (singleEngineContainer) singleEngineContainer.style.display = "none";
    if (comparisonContainer) comparisonContainer.style.display = "none";
    if (forensicAuditContainer) forensicAuditContainer.style.display = "block";

    // Target Meta Box
    if (videoMetaBox && targetFileName) {
      targetFileName.textContent = data.video_filename || (selectedFile ? selectedFile.name : "video.mp4");
      if (activeEngineLabel) {
        const engDisplay = engineName || (predData ? predData.engine : null);
        activeEngineLabel.textContent = engDisplay ? `${engDisplay} (Forensic Evidence Mode)` : "Forensic Multi-Scale Saliency Engine";
      }
      videoMetaBox.style.display = "flex";
    }

    // Verdict & Window Dispersion (Single-Run Provenance)
    const effThreshold = threshold !== null ? Number(threshold) : parseFloat(thresholdSlider.value || 0.50);

    // Maintain single-run provenance:
    // data.video_fake_probability is the spatiotemporal audit score, and data.confidence_interval is its window dispersion
    const auditProb = Number(data.video_fake_probability ?? 0);
    const isFake = auditProb >= effThreshold;
    const isUnderlyingInconclusive = (data.verdict === "INCONCLUSIVE") || (predData && predData.decision === "INCONCLUSIVE");

    const probPct = (auditProb * 100).toFixed(1);
    const ci = data.confidence_interval || [auditProb, auditProb];
    const ciLow = (Number(ci[0] ?? auditProb) * 100).toFixed(1);
    const ciHigh = (Number(ci[1] ?? auditProb) * 100).toFixed(1);

    // Update target meta box label with comparator engine context if present
    if (activeEngineLabel) {
      const engDisplay = engineName || (predData ? predData.engine : null) || "Forensic Multi-Scale Saliency Engine";
      let compNote = "";
      if (predData && predData.probability !== undefined) {
        const pScore = (Number(predData.probability) * 100).toFixed(1);
        const pDec = predData.decision || (Number(predData.probability) >= effThreshold ? "FAKE" : "REAL");
        compNote = ` • Selected Engine: ${pScore}% (${pDec})`;
      }
      activeEngineLabel.textContent = `${engDisplay} (Forensic Evidence Mode)${compNote}`;
    }

    if (forensicVerdictBadge) {
      if (isUnderlyingInconclusive) {
        forensicVerdictBadge.className = "forensic-verdict-badge inconclusive";
        if (forensicVerdictIcon) forensicVerdictIcon.textContent = "⚠️";
        if (forensicVerdictText) forensicVerdictText.textContent = "INCONCLUSIVE / INSUFFICIENT DATA";
      } else {
        forensicVerdictBadge.className = `forensic-verdict-badge ${isFake ? "fake" : "real"}`;
        if (forensicVerdictIcon) forensicVerdictIcon.textContent = isFake ? "🚨" : "🛡️";
        if (forensicVerdictText) forensicVerdictText.textContent = isFake ? "MANIPULATION DETECTED" : "NO MANIPULATION DETECTED";
      }
    }

    if (forensicCiPill) {
      forensicCiPill.textContent = (ciLow === ciHigh) ? `Window Score: ${ciLow}%` : `Window Dispersion: [${ciLow}% – ${ciHigh}%]`;
    }

    if (forensicProbVal) {
      forensicProbVal.textContent = `${probPct}%`;
    }

    if (forensicConfidenceLevel) {
      if (isUnderlyingInconclusive) {
        forensicConfidenceLevel.textContent = "Uncertain / Inconclusive";
        forensicConfidenceLevel.style.color = "#fbbf24";
      } else {
        const conf = data.verdict_confidence || (prob >= 0.85 || prob <= 0.15 ? "High Confidence" : "Moderate Confidence");
        forensicConfidenceLevel.textContent = conf;
        forensicConfidenceLevel.style.color = isFake ? "#f87171" : "#34d399";
      }
    }

    // Forensic Evidence Attribution Narrative
    if (forensicEvidenceBox) {
      forensicEvidenceBox.className = `forensic-evidence-box ${isUnderlyingInconclusive ? "inconclusive" : (isFake ? "fake" : "real")}`;
    }
    if (evidenceStrengthBadge) {
      if (isUnderlyingInconclusive) {
        evidenceStrengthBadge.textContent = "⚠️ INCONCLUSIVE AUDIT";
        evidenceStrengthBadge.style.color = "#fbbf24";
        evidenceStrengthBadge.style.background = "rgba(251, 191, 36, 0.15)";
      } else {
        evidenceStrengthBadge.textContent = isFake ? "🚨 ELEVATED ANOMALY INDICATORS" : "🛡️ NOMINAL BASELINE";
        evidenceStrengthBadge.style.color = isFake ? "#f87171" : "#34d399";
        evidenceStrengthBadge.style.background = isFake ? "rgba(239, 68, 68, 0.15)" : "rgba(52, 211, 153, 0.15)";
      }
    }

    // 1. Structured Technical Telemetry (Tab 2)
    if (forensicEvidenceText) {
      if (data.structured_telemetry) {
        renderStructuredTelemetry(forensicEvidenceText, data.structured_telemetry, data.forensic_explanation);
      } else {
        forensicEvidenceText.textContent = data.forensic_explanation || (isFake ? "Elevated manipulation indicators observed across temporal windows." : "No significant manipulation indicators detected within validated scope.");
      }
    }

    // 2. AI Forensic Analyst Report (Tab 1 - Human-Friendly Narrative)
    if (aiAnalystText) {
      if (data.groq_analyst_report && data._last_analyzed_threshold === effThreshold) {
        aiAnalystText.innerHTML = renderMarkdownToHtml(data.groq_analyst_report);
        if (refreshAiBtn) refreshAiBtn.style.display = "inline-block";
      } else {
        // Trigger automated Groq synthesis in background for this threshold
        fetchGroqSynthesis(data, predData, effThreshold);
      }
    }

    // Default to AI Analyst tab being active on audit render
    if (tabAiReport && tabRawTelemetry && aiAnalystBox && forensicEvidenceText) {
      tabAiReport.classList.add("active");
      tabRawTelemetry.classList.remove("active");
      aiAnalystBox.style.display = "block";
      forensicEvidenceText.style.display = "none";
    }

    // Forensic Quality & Biometric Stats
    if (forensicKinEnergy) {
      const kin = Number(data.kinematic_residual_energy ?? 0);
      forensicKinEnergy.textContent = kin.toFixed(2);
    }

    if (forensicTempStability) {
      const stab = Number(data.temporal_consistency_score ?? 1.0);
      forensicTempStability.textContent = `${(stab * 100).toFixed(1)}%`;
    }

    const windows = data.windows_analyzed || [];
    if (forensicWindowsCount) {
      forensicWindowsCount.textContent = `${windows.length} ${windows.length === 1 ? "Window" : "Windows"}`;
    }

    let totalFrames = 0;
    windows.forEach((w) => {
      totalFrames += Number(w.frame_count || 16);
    });
    if (forensicTotalFrames) {
      forensicTotalFrames.textContent = `${totalFrames} total frames`;
    }

    // 2D Spatial Saliency Heatmap
    if (data.saliency_heatmap_base64) {
      if (forensicSaliencyImg) {
        forensicSaliencyImg.src = `data:image/png;base64,${data.saliency_heatmap_base64}`;
        forensicSaliencyImg.style.display = "block";
      }
      if (forensicSaliencyPlaceholder) {
        forensicSaliencyPlaceholder.style.display = "none";
      }
    } else {
      if (forensicSaliencyImg) forensicSaliencyImg.style.display = "none";
      if (forensicSaliencyPlaceholder) forensicSaliencyPlaceholder.style.display = "flex";
    }

    // Warnings / Confounders
    const warnings = data.warnings || [];
    if (forensicWarningsBox && forensicWarningsList) {
      if (warnings.length > 0) {
        forensicWarningsList.innerHTML = "";
        warnings.forEach((warn) => {
          const li = document.createElement("li");
          li.textContent = warn;
          forensicWarningsList.appendChild(li);
        });
        forensicWarningsBox.style.display = "block";
      } else {
        forensicWarningsBox.style.display = "none";
      }
    }

    // Temporal Windows List
    if (forensicWindowsList) {
      forensicWindowsList.innerHTML = "";
      if (windows.length === 0) {
        const emptyDiv = document.createElement("div");
        emptyDiv.style.cssText = "color: var(--text-muted); font-size: 0.8rem; padding: 0.5rem 0;";
        emptyDiv.textContent = "No temporal face tracks detected.";
        forensicWindowsList.appendChild(emptyDiv);
      } else {
        windows.forEach((w) => {
          const card = document.createElement("div");
          card.className = "forensic-window-card";

          const wProb = Number(w.calibrated_fake_probability ?? w.raw_fake_probability ?? 0);
          const wIsFake = wProb >= effThreshold;
          const pillClass = wIsFake ? "fake" : "real";
          const wScorePct = (wProb * 100).toFixed(1);
          const startTime = Number(w.start_sec ?? 0).toFixed(1);
          const endTime = Number(w.end_sec ?? 0).toFixed(1);
          const faceConf = (Number(w.face_confidence ?? 1.0) * 100).toFixed(0);
          const kin = Number(w.kinematic_residual_energy ?? 0).toFixed(2);

          card.innerHTML = `
            <div class="window-meta">
              <div class="window-title">
                <span>Window #${w.window_id}</span>
                <span class="window-time-badge">${startTime}s – ${endTime}s</span>
              </div>
              <div style="font-size: 0.72rem; color: #94a3b8;">
                ${w.frame_count} frames • Face Track Conf: ${faceConf}% • Frame Differential: ${kin}
              </div>
            </div>
            <div class="window-metrics">
              <span class="window-score-pill ${pillClass}">${wIsFake ? "ELEVATED ANOMALY" : "NOMINAL"} (${wScorePct}%)</span>
              <span style="font-size: 0.68rem; font-family: var(--font-mono); color: var(--text-muted);">${w.processing_time_ms ? Number(w.processing_time_ms).toFixed(0) + " ms" : ""}</span>
            </div>
          `;
          forensicWindowsList.appendChild(card);
        });
      }
    }

    if (resetBtn) {
      resetBtn.style.display = "block";
    }
  }

  function renderComparison(data) {
    if (!data) return;

    if (singleEngineContainer) singleEngineContainer.style.display = "none";
    if (comparisonContainer) comparisonContainer.style.display = "block";
    if (forensicAuditContainer) forensicAuditContainer.style.display = "none";

    const consensus = data.consensus || "DISCORDANT";
    const signals = data.signals || {};
    const engines = data.engines || {};

    if (compConsensusBanner) {
      compConsensusBanner.className = "consensus-banner";
      if (consensus === "CONCORDANT_FAKE") {
        compConsensusBanner.classList.add("fake");
        if (compConsensusIcon) compConsensusIcon.textContent = "🚨";
        if (compConsensusTitle) compConsensusTitle.textContent = "CONSENSUS: CONCORDANT DEEPFAKE";
        if (compConsensusSubtitle) compConsensusSubtitle.textContent = "All models concordantly detect high-probability manipulation evidence across spatial and kinematic modalities.";
      } else if (consensus === "CONCORDANT_REAL") {
        compConsensusBanner.classList.add("real");
        if (compConsensusIcon) compConsensusIcon.textContent = "✅";
        if (compConsensusTitle) compConsensusTitle.textContent = "CONSENSUS: CONCORDANT AUTHENTIC";
        if (compConsensusSubtitle) compConsensusSubtitle.textContent = "All models concordantly detect authentic presentation within validated reference bounds.";
      } else if (consensus === "REJECT_OR_UNCERTAIN") {
        compConsensusBanner.classList.add("reject");
        if (compConsensusIcon) compConsensusIcon.textContent = "⚠️";
        if (compConsensusTitle) compConsensusTitle.textContent = "CONSENSUS: OPERATING ENVELOPE REJECT";
        if (compConsensusSubtitle) compConsensusSubtitle.textContent = "One or more engines declined attribution due to safety bounds or elevated predictive variance.";
      } else {
        compConsensusBanner.classList.add("discordant");
        if (compConsensusIcon) compConsensusIcon.textContent = "⚖️";
        if (compConsensusTitle) compConsensusTitle.textContent = "CONSENSUS: DISCORDANT (DOMAIN SPLIT)";
        if (compConsensusSubtitle) compConsensusSubtitle.textContent = "Divergence between static spatial features and temporal differential kinematics; arbitrated via M22 log-odds fusion.";
      }
    }

    if (compDiscordanceSummary) {
      compDiscordanceSummary.textContent = data.discordance_summary || "Multi-domain comparative assessment complete.";
    }

    const badgeClass = (d) => (d === "FAKE" ? "fake" : d === "REJECT" ? "reject" : "real");

    // Card 1: Champion M13.5
    const p13 = Number(signals.m13_5_probability ?? 0);
    const d13 = signals.decision_m13 || (p13 >= 0.5 ? "FAKE" : "REAL");
    if (cardVerdictM13) {
      cardVerdictM13.className = `engine-verdict-badge ${badgeClass(d13)}`;
      cardVerdictM13.textContent = `${d13} (p = ${p13.toFixed(4)})`;
    }
    if (cardFillM13) {
      cardFillM13.style.width = `${Math.min(100, Math.max(0, p13 * 100))}%`;
    }
    if (cardFreqM13) {
      const freq = (engines["v5-family-invariant"]?.metadata?.branch_scores?.p_freq) ?? signals.static_freq_score;
      cardFreqM13.textContent = freq !== undefined ? `p = ${Number(freq).toFixed(4)}` : "--";
    }
    if (cardTextureM13) {
      const tex = (engines["v5-family-invariant"]?.metadata?.branch_scores?.p_texture) ?? signals.static_texture_score;
      cardTextureM13.textContent = tex !== undefined ? `p = ${Number(tex).toFixed(4)}` : "--";
    }

    // Card 2: Baseline M18
    const p18 = Number(signals.m18_probability ?? 0);
    const d18 = signals.decision_m18 || (p18 >= 0.5 ? "FAKE" : "REAL");
    if (cardVerdictM18) {
      cardVerdictM18.className = `engine-verdict-badge ${badgeClass(d18)}`;
      cardVerdictM18.textContent = `${d18} (p = ${p18.toFixed(4)})`;
    }
    if (cardFillM18) {
      cardFillM18.style.width = `${Math.min(100, Math.max(0, p18 * 100))}%`;
    }
    if (cardKinM18) {
      const kin = (engines["m18-dual-stream"]?.metadata?.branch_scores?.p_kin) ?? signals.kinematic_motion_score;
      cardKinM18.textContent = kin !== undefined ? `p = ${Number(kin).toFixed(4)}` : "--";
    }
    if (cardGateM18) {
      const gate = (engines["m18-dual-stream"]?.metadata?.branch_scores?.gate_values) ?? signals.m18_gate_alpha;
      cardGateM18.textContent = gate !== undefined ? `α = ${Number(gate).toFixed(4)}` : "--";
    }

    // Card 3: Fused M22 Ensemble
    const p22 = Number(signals.m22_fused_probability ?? 0);
    const d22 = signals.decision_m22 || (p22 >= 0.9191 ? "FAKE" : "REAL");
    if (cardVerdictM22) {
      cardVerdictM22.className = `engine-verdict-badge ${badgeClass(d22)}`;
      cardVerdictM22.textContent = `${d22} (p = ${p22.toFixed(4)})`;
    }
    if (cardFillM22) {
      cardFillM22.style.width = `${Math.min(100, Math.max(0, p22 * 100))}%`;
    }
    if (cardLogitM22) {
      const raw = engines["m22-ensemble"]?.raw_logit;
      cardLogitM22.textContent = raw !== undefined ? `z = ${Number(raw).toFixed(3)} (p = ${p22.toFixed(4)})` : `p = ${p22.toFixed(4)}`;
    }

    // Target Meta Box
    if (videoMetaBox && targetFileName) {
      targetFileName.textContent = data.video_filename || (selectedFile ? selectedFile.name : "video.mp4");
      if (activeEngineLabel) {
        activeEngineLabel.textContent = "Multi-Engine Comparative Mode";
      }
      videoMetaBox.style.display = "flex";
    }

    if (resetBtn) {
      resetBtn.style.display = "block";
    }
  }

  function renderPrediction(data, activeThreshold) {
    if (!data) return;

    if (singleEngineContainer) singleEngineContainer.style.display = "block";
    if (comparisonContainer) comparisonContainer.style.display = "none";
    if (forensicAuditContainer) forensicAuditContainer.style.display = "none";

    const threshold = activeThreshold !== undefined ? activeThreshold : parseFloat(thresholdSlider.value);
    applyThresholdAndRenderVerdict(data, threshold);

    const prob = Number(data.probability ?? 0);
    const probPct = (prob * 100).toFixed(1);
    scoreVal.textContent = `${probPct}% (p = ${prob.toFixed(4)})`;
    scoreFill.style.width = `${Math.min(100, Math.max(0, parseFloat(probPct)))}%`;
    rawLogitVal.textContent = data.decision || (data.verdict === "Deepfake" ? "FAKE" : "REAL");

    framesVal.textContent = data.frames_analyzed ?? "--";
    facesVal.textContent = data.faces_detected ?? "--";

    // Branch Diagnostic Kinematic details
    const branchScores = data.metadata?.branch_scores || {};
    const ensembleSignals = data.metadata?.signals || {};
    const kinScore = branchScores.p_kin ?? ensembleSignals.kinematic_motion_score;
    const gateScore = branchScores.gate_values ?? ensembleSignals.m18_gate_alpha;

    if (kinScore !== undefined && techBranchKinRow && techBranchKinematic) {
      techBranchKinRow.style.display = "block";
      techBranchKinematic.textContent = `p = ${Number(kinScore).toFixed(4)}`;
    } else if (techBranchKinRow) {
      techBranchKinRow.style.display = "none";
    }

    if (gateScore !== undefined && techGateRow && techGateVal) {
      techGateRow.style.display = "block";
      techGateVal.textContent = `α = ${Number(gateScore).toFixed(4)}`;
    } else if (techGateRow) {
      techGateRow.style.display = "none";
    }

    // Latency Waterfall
    const lat = data.latency_ms || {};
    const prepMs = Number(lat.preprocessing_ms ?? 0);
    const modelMs = Number(lat.mc_dropout_inference_ms ?? lat.model_forward_ms ?? 0);
    const totalMs = Number(lat.total_ms ?? (prepMs + modelMs));

    prepLatencyVal.textContent = `${prepMs.toFixed(1)} ms`;
    modelLatencyVal.textContent = `${modelMs.toFixed(1)} ms`;
    totalLatencyVal.textContent = `${totalMs.toFixed(1)} ms`;

    const prepPct = totalMs > 0 ? (prepMs / totalMs) * 100 : 70;
    const modelPct = totalMs > 0 ? (modelMs / totalMs) * 100 : 30;

    prepLatencyBar.style.width = `${Math.min(100, Math.max(0, prepPct))}%`;
    modelLatencyBar.style.width = `${Math.min(100, Math.max(0, modelPct))}%`;

    // Video Target Meta Box
    if (videoMetaBox && targetFileName) {
      targetFileName.textContent = data.original_filename || (selectedFile ? selectedFile.name : "video.mp4");
      if (activeEngineLabel) {
        const eng = data.engine || (engineSelect ? engineSelect.value : "v5-family-invariant");
        activeEngineLabel.textContent = eng;
      }
      videoMetaBox.style.display = "flex";
    }

    // Show Analyze Another Video button
    if (resetBtn) {
      resetBtn.style.display = "block";
    }
  }

  // 6. Reset / Analyze Another Video
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      selectedFile = null;
      lastPredictionData = null;
      lastAuditData = null;
      fileInput.value = "";
      videoPreview.src = "";
      previewContainer.style.display = "none";
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Select Video to Analyze";
      if (compareBtn) {
        compareBtn.disabled = true;
        compareBtn.textContent = "⚖️ Compare All Engines";
      }

      thresholdSlider.value = "0.50";
      thresholdVal.textContent = "0.50";

      verdictBox.className = "verdict-box";
      verdictText.textContent = "AWAITING INPUT";
      verdictConfidence.textContent = "Model Score: --";

      if (rejectAlertBanner) rejectAlertBanner.style.display = "none";
      if (diagnosticContainer) diagnosticContainer.style.display = "none";
      if (decisionReason) decisionReason.style.display = "none";
      if (techBranchContainer) techBranchContainer.style.display = "none";
      if (techBranchKinRow) techBranchKinRow.style.display = "none";
      if (techGateRow) techGateRow.style.display = "none";

      if (comparisonContainer) comparisonContainer.style.display = "none";
      if (forensicAuditContainer) forensicAuditContainer.style.display = "none";
      if (singleEngineContainer) singleEngineContainer.style.display = "block";

      if (forensicSaliencyImg) {
        forensicSaliencyImg.src = "";
        forensicSaliencyImg.style.display = "none";
      }
      if (forensicSaliencyPlaceholder) {
        forensicSaliencyPlaceholder.style.display = "flex";
      }
      if (forensicWarningsBox) {
        forensicWarningsBox.style.display = "none";
      }
      if (forensicWindowsList) {
        forensicWindowsList.innerHTML = "";
      }
      if (forensicEvidenceText) {
        forensicEvidenceText.textContent = "Awaiting video inference to generate forensic attribution evidence...";
      }
      if (tabAiReport && tabRawTelemetry) {
        tabAiReport.classList.add("active");
        tabRawTelemetry.classList.remove("active");
      }
      if (aiAnalystBox) {
        aiAnalystBox.style.display = "block";
      }
      if (aiAnalystText) {
        aiAnalystText.innerHTML = `
          <div style="color: var(--text-muted); font-size: 0.8rem; padding: 0.5rem 0;">
            Awaiting video inference to generate AI forensic synthesis...
          </div>
        `;
      }
      if (refreshAiBtn) {
        refreshAiBtn.style.display = "none";
      }
      if (forensicEvidenceBox) {
        forensicEvidenceBox.className = "forensic-evidence-box";
      }
      if (evidenceStrengthBadge) {
        evidenceStrengthBadge.textContent = "Physical Anomaly Analysis";
        evidenceStrengthBadge.style.color = "#94a3b8";
        evidenceStrengthBadge.style.background = "rgba(255, 255, 255, 0.08)";
      }

      scoreVal.textContent = "--%";
      scoreFill.style.width = "0%";
      rawLogitVal.textContent = "Calibrated";
      framesVal.textContent = "--";
      facesVal.textContent = "--";

      prepLatencyVal.textContent = "-- ms";
      modelLatencyVal.textContent = "-- ms";
      totalLatencyVal.textContent = "-- ms";
      prepLatencyBar.style.width = "0%";
      modelLatencyBar.style.width = "0%";

      if (videoMetaBox) videoMetaBox.style.display = "none";
      resetBtn.style.display = "none";
    });
  }
});

