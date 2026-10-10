import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost";
  size?: "sm" | "md" | "lg";
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", disabled, ...props }, ref) => (
    <button
      ref={ref}
      disabled={disabled}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        {
          "bg-primary text-white hover:bg-primary/90 shadow-sm hover:shadow-md":
            variant === "primary",
          "bg-secondary text-white hover:bg-secondary/90 shadow-sm":
            variant === "secondary",
          "border border-border bg-white text-secondary hover:bg-gray-50":
            variant === "outline",
          "text-secondary hover:bg-gray-100": variant === "ghost",
        },
        {
          "h-9 px-3 text-sm": size === "sm",
          "h-10 px-4 text-sm": size === "md",
          "h-12 px-6 text-base": size === "lg",
        },
        className
      )}
      {...props}
    />
  )
);
Button.displayName = "Button";

export { Button };
