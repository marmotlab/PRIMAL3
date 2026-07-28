const zoomableVideos = document.querySelectorAll("[data-video-zoom]");

zoomableVideos.forEach((card) => {
  const viewport = card.querySelector(".simulation-card__viewport");
  const video = viewport.querySelector("video");
  const lens = viewport.querySelector(".simulation-card__lens");
  const context = lens.getContext("2d");
  const zoomLevel = 2.35;
  let pointerX = 0;
  let pointerY = 0;
  let animationFrame;

  const positionLens = (event) => {
    const bounds = viewport.getBoundingClientRect();
    const lensRadius = lens.getBoundingClientRect().width / 2;
    const edgeGap = 7;

    pointerX = Math.max(0, Math.min(bounds.width, event.clientX - bounds.left));
    pointerY = Math.max(0, Math.min(bounds.height, event.clientY - bounds.top));

    const lensX = Math.max(lensRadius + edgeGap, Math.min(bounds.width - lensRadius - edgeGap, pointerX));
    const lensY = Math.max(lensRadius + edgeGap, Math.min(bounds.height - lensRadius - edgeGap, pointerY));

    lens.style.left = `${lensX}px`;
    lens.style.top = `${lensY}px`;
  };

  const drawLens = () => {
    if (!card.classList.contains("is-zooming")) return;

    if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
      const viewportBounds = viewport.getBoundingClientRect();
      const lensBounds = lens.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const canvasSize = Math.round(lensBounds.width * pixelRatio);

      if (lens.width !== canvasSize || lens.height !== canvasSize) {
        lens.width = canvasSize;
        lens.height = canvasSize;
      }

      const coverScale = Math.max(
        viewportBounds.width / video.videoWidth,
        viewportBounds.height / video.videoHeight,
      );
      const cropX = (video.videoWidth * coverScale - viewportBounds.width) / 2;
      const cropY = (video.videoHeight * coverScale - viewportBounds.height) / 2;
      const centerX = (pointerX + cropX) / coverScale;
      const centerY = (pointerY + cropY) / coverScale;
      const sourceSize = lensBounds.width / zoomLevel / coverScale;
      const sourceX = Math.max(0, Math.min(video.videoWidth - sourceSize, centerX - sourceSize / 2));
      const sourceY = Math.max(0, Math.min(video.videoHeight - sourceSize, centerY - sourceSize / 2));

      context.clearRect(0, 0, lens.width, lens.height);
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(
        video,
        sourceX,
        sourceY,
        sourceSize,
        sourceSize,
        0,
        0,
        lens.width,
        lens.height,
      );
    }

    animationFrame = window.requestAnimationFrame(drawLens);
  };

  viewport.addEventListener("pointerenter", (event) => {
    if (event.pointerType !== "touch") {
      positionLens(event);
      card.classList.add("is-zooming");
      window.cancelAnimationFrame(animationFrame);
      animationFrame = window.requestAnimationFrame(drawLens);
    }
  });

  viewport.addEventListener("pointermove", (event) => {
    if (event.pointerType === "touch") return;
    positionLens(event);
  });

  viewport.addEventListener("pointerleave", () => {
    card.classList.remove("is-zooming");
    window.cancelAnimationFrame(animationFrame);
  });
});
