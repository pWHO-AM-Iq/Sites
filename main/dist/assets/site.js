"use strict";
document.querySelectorAll("[data-overlay]").forEach(button => {
  button.addEventListener("click", () => {
    const enabled = button.getAttribute("aria-pressed") !== "true";
    button.setAttribute("aria-pressed", String(enabled));
    button.querySelector("b").textContent = enabled ? "ON" : "OFF";
    button.closest("figure").querySelector(".boxes").hidden = !enabled;
  });
});
document.addEventListener("keydown", event => {
  const menu = document.querySelector(".menu[open]");
  if (event.key === "Escape" && menu) {
    menu.open = false;
    menu.querySelector("summary").focus();
  }
});
document.addEventListener("click", event => {
  const menu = document.querySelector(".menu[open]");
  if (menu && !menu.contains(event.target)) menu.open = false;
});
