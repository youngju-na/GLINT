/* ─── Media compare (video & image sliders) ───────── */
function initMediaCompare(compareRoot) {
  const overlay = compareRoot.querySelector(".compare-overlay");
  const divider = compareRoot.querySelector(".compare-divider");
  const handle = compareRoot.querySelector(".compare-handle");
  const videos = compareRoot.querySelectorAll("video");
  const initial = Number(compareRoot.dataset.divider || "0.5");

  let activePointer = false;

  function setRatio(ratio) {
    const clamped = Math.min(Math.max(ratio, 0), 1);
    overlay.style.clipPath = `inset(0 ${100 - clamped * 100}% 0 0)`;
    divider.style.left = `${clamped * 100}%`;
  }

  function updateFromClientX(clientX) {
    const rect = compareRoot.getBoundingClientRect();
    setRatio((clientX - rect.left) / rect.width);
  }

  function syncVideos(source) {
    videos.forEach((video) => {
      if (video === source) return;
      if (Math.abs(video.currentTime - source.currentTime) > 0.08) {
        video.currentTime = source.currentTime;
      }
    });
  }

  setRatio(initial);

  /* Drag hint animation on first view */
  if (handle) {
    const hintObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setTimeout(() => handle.classList.add("hint-animate"), 600);
            handle.addEventListener("animationend", () => handle.classList.remove("hint-animate"), { once: true });
            hintObserver.disconnect();
          }
        });
      },
      { threshold: 0.5 }
    );
    hintObserver.observe(compareRoot);
  }

  compareRoot.addEventListener("pointerdown", (event) => {
    activePointer = true;
    compareRoot.setPointerCapture(event.pointerId);
    updateFromClientX(event.clientX);
  });

  compareRoot.addEventListener("pointermove", (event) => {
    if (!activePointer) return;
    updateFromClientX(event.clientX);
  });

  compareRoot.addEventListener("pointerup", () => {
    activePointer = false;
  });

  compareRoot.addEventListener("pointerleave", () => {
    activePointer = false;
  });

  compareRoot.addEventListener("mousemove", (event) => {
    if (activePointer) return;
    updateFromClientX(event.clientX);
  });

  compareRoot.addEventListener(
    "touchmove",
    (event) => {
      if (!event.touches[0]) return;
      updateFromClientX(event.touches[0].clientX);
    },
    { passive: true }
  );

  if (videos.length) {
    videos.forEach((video) => {
      video.addEventListener("play", () => {
        videos.forEach((other) => {
          if (other !== video && other.paused) other.play().catch(() => {});
        });
      });

      video.addEventListener("pause", () => {
        videos.forEach((other) => {
          if (other !== video && !other.paused) other.pause();
        });
      });

      video.addEventListener("timeupdate", () => syncVideos(video));
    });
  }
}

/* ─── Surface tabs ────────────────────────────────── */
function initSurfaceTabs() {
  const tabs = document.querySelectorAll(".surface-tab");
  const panels = document.querySelectorAll(".surface-panel");

  if (!tabs.length || !panels.length) return;

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.surfaceTarget;

      tabs.forEach((item) => item.classList.toggle("is-active", item === tab));
      panels.forEach((panel) => {
        panel.classList.toggle("is-active", panel.dataset.surfacePanel === target);
      });
    });
  });
}

/* ─── Surface baselines ───────────────────────────── */
function initSurfaceBaselines() {
  document.querySelectorAll(".surface-panel").forEach((panel) => {
    const buttons = panel.querySelectorAll(".surface-baseline-tab");
    const baselineImage = panel.querySelector(".compare-image-baseline");
    const baselineLabel = panel.querySelector(".js-baseline-label");

    if (!buttons.length || !baselineImage || !baselineLabel) return;

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        buttons.forEach((item) => item.classList.toggle("is-active", item === button));
        baselineImage.src = button.dataset.baselineSrc;
        baselineLabel.textContent = button.dataset.baselineName;
        baselineImage.alt = `${button.dataset.baselineName} surface reconstruction mesh.`;
      });
    });
  });
}

/* ─── Video carousel ──────────────────────────────── */
function initVideoCarousel(carousel) {
  const track = carousel.querySelector(".video-carousel-track");
  const cards = carousel.querySelectorAll(".video-scene-card");
  const prev = carousel.querySelector(".carousel-arrow-prev");
  const next = carousel.querySelector(".carousel-arrow-next");
  const desktopVisible = Number(carousel.dataset.visibleDesktop || "2");
  const mobileVisible = Number(carousel.dataset.visibleMobile || "1");
  let index = 0;
  let autoTimer = null;
  let isPaused = false;

  function visibleCount() {
    return window.innerWidth <= 768 ? mobileVisible : desktopVisible;
  }

  /* Build dot indicators — insert at the top of the carousel */
  let dotsContainer = carousel.querySelector(".carousel-dots");
  if (!dotsContainer) {
    dotsContainer = document.createElement("div");
    dotsContainer.className = "carousel-dots";
    carousel.insertBefore(dotsContainer, carousel.firstChild);
  }

  function buildDots() {
    dotsContainer.innerHTML = "";
    const show = visibleCount();
    const total = Math.max(cards.length - show + 1, 1);
    for (let i = 0; i < total; i++) {
      const dot = document.createElement("button");
      dot.className = "carousel-dot" + (i === index ? " is-active" : "");
      dot.type = "button";
      dot.setAttribute("aria-label", `Go to slide ${i + 1}`);
      dot.addEventListener("click", () => {
        index = i;
        update();
        resetAutoAdvance();
      });
      dotsContainer.appendChild(dot);
    }
  }

  function updateDots() {
    const dots = dotsContainer.querySelectorAll(".carousel-dot");
    dots.forEach((dot, i) => dot.classList.toggle("is-active", i === index));
  }

  function update() {
    const show = visibleCount();
    const maxIndex = Math.max(cards.length - show, 0);
    index = Math.min(Math.max(index, 0), maxIndex);
    const offset = (100 / show) * index;
    track.style.transform = `translateX(-${offset}%)`;
    prev.disabled = index <= 0;
    next.disabled = index >= maxIndex;
    updateDots();
  }

  function autoAdvance() {
    if (isPaused) return;
    const show = visibleCount();
    const maxIndex = Math.max(cards.length - show, 0);
    index = index >= maxIndex ? 0 : index + 1;
    update();
  }

  function resetAutoAdvance() {
    clearInterval(autoTimer);
    autoTimer = setInterval(autoAdvance, 5000);
  }

  prev.addEventListener("click", () => {
    index -= 1;
    update();
    resetAutoAdvance();
  });

  next.addEventListener("click", () => {
    index += 1;
    update();
    resetAutoAdvance();
  });

  carousel.addEventListener("mouseenter", () => { isPaused = true; });
  carousel.addEventListener("mouseleave", () => { isPaused = false; });

  window.addEventListener("resize", () => {
    buildDots();
    update();
  });

  buildDots();
  update();
  resetAutoAdvance();
}

/* ─── Scroll reveal ───────────────────────────────── */
function initScrollReveal() {
  const revealElements = document.querySelectorAll(".reveal, .reveal-scale");
  if (!revealElements.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  revealElements.forEach((el) => observer.observe(el));
}

/* ─── Metric count-up animation ───────────────────── */
function initMetricCountUp() {
  const metricNumbers = document.querySelectorAll(".metric-number[data-target]");
  if (!metricNumbers.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;

        const el = entry.target;
        if (el.dataset.animated) return;
        el.dataset.animated = "true";

        const target = parseFloat(el.dataset.target);
        const decimals = (el.dataset.target.split(".")[1] || "").length;
        const duration = 1200;
        const startTime = performance.now();

        function tick(now) {
          const elapsed = now - startTime;
          const progress = Math.min(elapsed / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          const current = eased * target;
          el.textContent = current.toFixed(decimals);
          if (progress < 1) requestAnimationFrame(tick);
        }

        requestAnimationFrame(tick);

        /* Animate the sibling bar if present */
        const barWrap = el.closest(".metric-item, .metric-box");
        if (barWrap) {
          const bar = barWrap.querySelector(".metric-bar");
          if (bar) {
            const percent = bar.dataset.percent || "0";
            setTimeout(() => { bar.style.width = percent + "%"; }, 100);
          }
        }

        observer.unobserve(el);
      });
    },
    { threshold: 0.3 }
  );

  metricNumbers.forEach((el) => observer.observe(el));
}

/* ─── Dark-mode toggle ────────────────────────────── */
function initThemeToggle() {
  const toggle = document.querySelector(".theme-toggle");
  if (!toggle) return;

  const icon = toggle.querySelector(".theme-toggle-icon");
  const label = toggle.querySelector(".theme-toggle-label");
  const root = document.documentElement;

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    if (icon) icon.textContent = theme === "dark" ? "\u263E" : "\u2600";
    if (label) label.textContent = theme === "dark" ? "Light" : "Dark";
    try { localStorage.setItem("glint-theme", theme); } catch (e) { /* noop */ }
  }

  /* Restore saved or system preference */
  const saved = (() => { try { return localStorage.getItem("glint-theme"); } catch (e) { return null; } })();
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(saved || (prefersDark ? "dark" : "light"));

  toggle.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
  });
}

/* ─── Citation copy ───────────────────────────────── */
function initCitationCopy() {
  const btn = document.querySelector(".citation-copy-btn");
  if (!btn) return;

  btn.addEventListener("click", () => {
    const block = document.querySelector(".citation-block code");
    if (!block) return;

    navigator.clipboard.writeText(block.textContent.trim()).then(() => {
      btn.classList.add("is-copied");
      const originalText = btn.querySelector(".copy-label");
      if (originalText) originalText.textContent = "Copied!";
      setTimeout(() => {
        btn.classList.remove("is-copied");
        if (originalText) originalText.textContent = "Copy";
      }, 2000);
    }).catch(() => {});
  });
}

/* ─── Boot ────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".video-compare, .image-compare").forEach(initMediaCompare);
  document.querySelectorAll(".video-carousel").forEach(initVideoCarousel);
  initSurfaceTabs();
  initSurfaceBaselines();
  initScrollReveal();
  initMetricCountUp();
  initThemeToggle();
  initCitationCopy();
});
