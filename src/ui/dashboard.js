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

  // Forensic Audit Elements
  const forensicAuditBtn = document.getElementById("forensicAuditBtn");
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

  // 1. Fetch Health Status
  fetch("/health")
    .then((res) => res.json())
    .then((data) => {
      if (data.status === "healthy") {
        healthDot.classList.add("online");
        healthText.textContent = `API Ready (${data.available_engines.join(", ")})`;
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

  thresholdSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    thresholdVal.textContent = val.toFixed(2);
    if (lastPredictionData) {
      applyThresholdAndRenderVerdict(lastPredictionData, val);
    }
  });

  if (resetThresholdBtn) {
    resetThresholdBtn.addEventListener("click", () => {
      thresholdSlider.value = "0.50";
      thresholdVal.textContent = "0.50";
      if (lastPredictionData) {
        applyThresholdAndRenderVerdict(lastPredictionData, 0.50);
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
    if (forensicAuditBtn) {
      forensicAuditBtn.disabled = false;
      forensicAuditBtn.textContent = "🔬 Forensic Audit";
    }

    // Show local video preview
    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    previewContainer.style.display = "flex";
  }

  // 5. Submit Video for Single-Engine Inference
  analyzeBtn.addEventListener("click", () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Processing Video (Sampling, MTCNN & MC-Dropout)...";
    if (compareBtn) compareBtn.disabled = true;
    if (forensicAuditBtn) forensicAuditBtn.disabled = true;

    const activeEngine = engineSelect ? engineSelect.value : "v5-family-invariant";
    const currentThreshold = parseFloat(thresholdSlider.value);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("threshold", currentThreshold);
    formData.append("engine", activeEngine);
    formData.append("calibrated", "true");

    fetch("/api/v1/predict", {
      method: "POST",
      body: formData,
    })
      .then((res) => {
        if (!res.ok) {
          return res.json().then((err) => {
            throw new Error(err.detail || "Inference failed");
          });
        }
        return res.json();
      })
      .then((data) => {
        lastPredictionData = data;
        renderPrediction(data, currentThreshold);
      })
      .catch((err) => {
        alert(`Inference Error: ${err.message}`);
      })
      .finally(() => {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = `Analyze ${selectedFile.name}`;
        if (compareBtn) compareBtn.disabled = false;
        if (forensicAuditBtn) forensicAuditBtn.disabled = false;
      });
  });

  // 5b. Submit Video for Multi-Engine Comparison
  if (compareBtn) {
    compareBtn.addEventListener("click", () => {
      if (!selectedFile) return;

      compareBtn.disabled = true;
      compareBtn.textContent = "Comparing Engines (M13.5 + M18 + M22)...";
      if (analyzeBtn) analyzeBtn.disabled = true;
      if (forensicAuditBtn) forensicAuditBtn.disabled = true;

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
          if (forensicAuditBtn) forensicAuditBtn.disabled = false;
        });
    });
  }

  // 5c. Submit Video for Spatiotemporal Forensic Audit
  if (forensicAuditBtn) {
    forensicAuditBtn.addEventListener("click", () => {
      if (!selectedFile) return;

      forensicAuditBtn.disabled = true;
      forensicAuditBtn.textContent = "Auditing (Laplacian, Multi-Window)...";
      if (analyzeBtn) analyzeBtn.disabled = true;
      if (compareBtn) compareBtn.disabled = true;

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("windows", "4");
      formData.append("frames_per_window", "16");

      fetch("/api/v1/forensic-audit", {
        method: "POST",
        body: formData,
      })
        .then((res) => {
          if (!res.ok) {
            return res.json().then((err) => {
              throw new Error(err.detail || "Forensic audit failed");
            });
          }
          return res.json();
        })
        .then((data) => {
          renderForensicAudit(data);
        })
        .catch((err) => {
          alert(`Forensic Audit Error: ${err.message}`);
        })
        .finally(() => {
          forensicAuditBtn.disabled = false;
          forensicAuditBtn.textContent = "🔬 Forensic Audit";
          if (analyzeBtn) analyzeBtn.disabled = false;
          if (compareBtn) compareBtn.disabled = false;
        });
    });
  }

  function renderForensicAudit(data) {
    if (!data) return;

    if (singleEngineContainer) singleEngineContainer.style.display = "none";
    if (comparisonContainer) comparisonContainer.style.display = "none";
    if (forensicAuditContainer) forensicAuditContainer.style.display = "block";

    // Target Meta Box
    if (videoMetaBox && targetFileName) {
      targetFileName.textContent = data.video_filename || (selectedFile ? selectedFile.name : "video.mp4");
      if (activeEngineLabel) {
        activeEngineLabel.textContent = "Forensic Multi-Scale Saliency Engine";
      }
      videoMetaBox.style.display = "flex";
    }

    // Verdict & CI
    const verdict = (data.verdict || "AUTHENTIC").toUpperCase();
    const prob = Number(data.video_fake_probability ?? 0);
    const probPct = (prob * 100).toFixed(1);
    const ci = data.confidence_interval || [prob, prob];
    const ciLow = (Number(ci[0] ?? prob) * 100).toFixed(1);
    const ciHigh = (Number(ci[1] ?? prob) * 100).toFixed(1);

    if (forensicVerdictBadge) {
      forensicVerdictBadge.className = "forensic-verdict-badge";
      if (verdict.includes("DEEPFAKE") || verdict.includes("FAKE")) {
        forensicVerdictBadge.classList.add("fake");
        if (forensicVerdictIcon) forensicVerdictIcon.textContent = "🚨";
        if (forensicVerdictText) forensicVerdictText.textContent = "DEEPFAKE DETECTED";
      } else if (verdict.includes("INCONCLUSIVE") || verdict.includes("SUSPICIOUS") || verdict.includes("REJECT")) {
        forensicVerdictBadge.classList.add("inconclusive");
        if (forensicVerdictIcon) forensicVerdictIcon.textContent = "⚠️";
        if (forensicVerdictText) forensicVerdictText.textContent = verdict;
      } else {
        forensicVerdictBadge.classList.add("real");
        if (forensicVerdictIcon) forensicVerdictIcon.textContent = "🛡️";
        if (forensicVerdictText) forensicVerdictText.textContent = "AUTHENTIC VIDEO";
      }
    }

    if (forensicCiPill) {
      forensicCiPill.textContent = `95% CI: [${ciLow}% – ${ciHigh}%]`;
    }

    if (forensicProbVal) {
      forensicProbVal.textContent = `${probPct}%`;
    }

    if (forensicConfidenceLevel) {
      const conf = data.verdict_confidence || "High Confidence";
      forensicConfidenceLevel.textContent = conf;
      if (conf.toLowerCase().includes("high")) {
        forensicConfidenceLevel.style.color = "#34d399";
      } else if (conf.toLowerCase().includes("mod") || conf.toLowerCase().includes("med")) {
        forensicConfidenceLevel.style.color = "#fbbf24";
      } else {
        forensicConfidenceLevel.style.color = "#f87171";
      }
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
          const isFake = wProb >= 0.5;
          const pillClass = isFake ? "fake" : "real";
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
                ${w.frame_count} frames • Face Track Conf: ${faceConf}% • Kinematic Energy: ${kin}
              </div>
            </div>
            <div class="window-metrics">
              <span class="window-score-pill ${pillClass}">${isFake ? "MANIPULATION" : "NATURAL"} (${wScorePct}%)</span>
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
      fileInput.value = "";
      videoPreview.src = "";
      previewContainer.style.display = "none";
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Select Video to Analyze";
      if (compareBtn) {
        compareBtn.disabled = true;
        compareBtn.textContent = "⚖️ Compare All Engines";
      }
      if (forensicAuditBtn) {
        forensicAuditBtn.disabled = true;
        forensicAuditBtn.textContent = "🔬 Forensic Audit";
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

