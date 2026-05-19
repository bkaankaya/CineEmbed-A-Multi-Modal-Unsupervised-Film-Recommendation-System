"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Film, Sparkles, Boxes, Images, BookOpen } from "lucide-react";

const NAV = [
  { href: "/", label: "Home", icon: Sparkles },
  { href: "/cluster", label: "Clusters", icon: Boxes },
  { href: "/gallery", label: "Gallery", icon: Images },
  { href: "/about", label: "About", icon: BookOpen },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <nav
      aria-label="Primary navigation"
      className="fixed left-0 top-0 bottom-0 w-[220px] bg-card border-r border-border p-4 overflow-hidden"
    >
      {/* V2 — decorative vertical accent strip */}
      <div
        aria-hidden="true"
        className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-purple-200/60 via-purple-100/20 to-transparent pointer-events-none"
      />

      <h1 className="font-semibold mb-6 text-purple-700 flex items-center gap-2 relative">
        {/* V2 — radial glow behind logo icon */}
        <span className="relative inline-flex">
          <span
            aria-hidden="true"
            className="absolute -inset-2 bg-purple-200/40 blur-xl rounded-full pointer-events-none"
          />
          <Film className="w-5 h-5 relative" aria-hidden="true" />
        </span>
        <span className="relative">CineEmbed</span>
      </h1>
      <ul className="space-y-1 relative">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || (href !== "/" && pathname.startsWith(href));
          return (
            <li key={href}>
              <Link
                href={href}
                aria-current={active ? "page" : undefined}
                className={`flex items-center gap-2 py-1.5 rounded text-sm ${
                  active
                    ? "bg-purple-50 text-purple-800 border-l-2 border-primary pl-[6px] pr-2 font-semibold"
                    : "text-gray-700 hover:bg-gray-50 px-2"
                }`}
              >
                <Icon className="w-4 h-4" aria-hidden="true" />
                {label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
