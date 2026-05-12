document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("mobile-menu-button");
    const menu = document.getElementById("mobile-menu");
    if (toggle && menu) {
        toggle.addEventListener("click", () => {
            menu.classList.toggle("is-open");
        });
    }

    const body = document.body;
    const desktopThemeToggle = document.getElementById("theme-toggle-desktop");
    const mobileThemeToggle = document.getElementById("theme-toggle-mobile");
    const allThemeToggles = [desktopThemeToggle, mobileThemeToggle].filter(Boolean);

    const applyTheme = (theme) => {
        const useDark = theme === "dark";
        body.classList.toggle("dark-mode", useDark);
        allThemeToggles.forEach((el) => {
            el.checked = useDark;
        });
    };

    let storedTheme = "light";
    try {
        storedTheme = localStorage.getItem("theme") || "light";
    } catch (error) {
        storedTheme = "light";
    }
    applyTheme(storedTheme);

    allThemeToggles.forEach((el) => {
        el.addEventListener("change", () => {
            const nextTheme = el.checked ? "dark" : "light";
            applyTheme(nextTheme);
            try {
                localStorage.setItem("theme", nextTheme);
            } catch (error) {
                // no-op when storage is unavailable
            }
        });
    });

    const userButton = document.getElementById("user-menu-button");
    const userPanel = document.getElementById("user-menu-panel");
    if (userButton && userPanel) {
        userButton.addEventListener("click", (event) => {
            event.stopPropagation();
            userPanel.classList.toggle("hidden");
        });

        document.addEventListener("click", (event) => {
            if (!userPanel.classList.contains("hidden") && !userPanel.contains(event.target) && event.target !== userButton) {
                userPanel.classList.add("hidden");
            }
        });
    }
});
