import { Brain, Menu, User, FileText, Upload, Home } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

interface TopNavigationProps {
  onMenuToggle?: () => void;
}

export const TopNavigation = ({ onMenuToggle }: TopNavigationProps) => {
  return (
    <header className="h-16 bg-card border-b border-border flex items-center justify-between px-6 shadow-sm relative z-50">
      {/* Logo & App Name */}
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={onMenuToggle}
          className="md:hidden"
        >
          <Menu className="h-5 w-5" />
        </Button>
        
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">NeuroInsight AI</h1>
            <p className="text-xs text-muted-foreground">Medical Imaging Analytics</p>
          </div>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="hidden md:flex items-center gap-1">
        <Button variant="ghost" className="gap-2">
          <Home className="h-4 w-4" />
          Home
        </Button>
        <Button variant="ghost" className="gap-2">
          <Upload className="h-4 w-4" />
          Upload
        </Button>
        <Button variant="ghost" className="gap-2">
          <FileText className="h-4 w-4" />
          Documentation
        </Button>
      </nav>

      {/* User Section */}
      <div className="flex items-center gap-3">
        <Avatar className="h-8 w-8">
          <AvatarFallback className="bg-primary text-primary-foreground">
            <User className="h-4 w-4" />
          </AvatarFallback>
        </Avatar>
      </div>
    </header>
  );
};