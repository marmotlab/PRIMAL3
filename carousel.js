const carousel = document.querySelector("[data-carousel]");

if (carousel) {
  const slides = Array.from(carousel.querySelectorAll("[data-slide]"));
  const dots = Array.from(carousel.querySelectorAll("[data-carousel-dot]"));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const intervalDuration = 3200;
  let activeIndex = 0;
  let intervalId;

  const showSlide = (nextIndex) => {
    activeIndex = (nextIndex + slides.length) % slides.length;

    slides.forEach((slide, index) => {
      const isActive = index === activeIndex;
      slide.classList.toggle("is-active", isActive);
      slide.setAttribute("aria-hidden", String(!isActive));
    });

    dots.forEach((dot, index) => {
      const isActive = index === activeIndex;
      dot.classList.toggle("is-active", isActive);
      dot.setAttribute("aria-current", String(isActive));
    });
  };

  const stopRotation = () => {
    window.clearInterval(intervalId);
    intervalId = undefined;
  };

  const startRotation = () => {
    stopRotation();

    if (!reduceMotion.matches && !document.hidden) {
      intervalId = window.setInterval(() => showSlide(activeIndex + 1), intervalDuration);
    }
  };

  dots.forEach((dot) => {
    dot.addEventListener("click", () => {
      showSlide(Number(dot.dataset.carouselDot));
      startRotation();
    });
  });

  carousel.addEventListener("mouseenter", stopRotation);
  carousel.addEventListener("mouseleave", startRotation);
  carousel.addEventListener("focusin", stopRotation);
  carousel.addEventListener("focusout", (event) => {
    if (!carousel.contains(event.relatedTarget)) {
      startRotation();
    }
  });

  document.addEventListener("visibilitychange", startRotation);
  reduceMotion.addEventListener("change", startRotation);
  startRotation();
}
