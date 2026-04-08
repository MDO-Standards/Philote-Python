import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
    docsSidebar: [
        {
            type: "category",
            label: "Getting Started",
            items: [
                "getting-started/installation",
                "getting-started/quickstart",
            ],
        },
        {
            type: "category",
            label: "Tutorials",
            items: [
                "tutorials/explicit-disciplines",
                "tutorials/implicit-disciplines",
                "tutorials/units",
            ],
        },
        {
            type: "category",
            label: "Working with OpenMDAO",
            items: [
                "openmdao/openmdao-clients",
                "openmdao/openmdao-groups",
            ],
        },
        {
            type: "category",
            label: "About",
            items: ["about/license"],
        },
    ],
};

export default sidebars;
