import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Activity, 
  Clock, 
  User, 
  Cpu, 
  Database,
  Wifi
} from "lucide-react";

export const StatusBar = ({ analyzing = false, progress = 0 }: { analyzing?: boolean; progress?: number }) => {
  return (
    <div className="h-12 bg-card border-t border-border px-6 flex items-center justify-between text-sm">
      {/* Left Section - Patient Info */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <User className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">Patient:</span>
          <Badge variant="outline">ANON-001</Badge>
        </div>
        
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">Study:</span>
          <span className="font-mono text-xs">2024-09-14_15:42:33</span>
        </div>
      </div>

      {/* Center Section - Processing Status */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          {analyzing ? (
            <>
              <Activity className="h-4 w-4 text-primary animate-pulse" />
              <span className="text-primary font-medium">Analyzing…</span>
              <div className="w-40">
                <Progress value={progress} />
              </div>
              <span className="tabular-nums w-10 text-right">{Math.floor(progress)}%</span>
            </>
          ) : (
            <>
              <Activity className="h-4 w-4 text-success" />
              <span className="text-success font-medium">Ready</span>
            </>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">GPU:</span>
          <Badge variant="secondary">Available</Badge>
        </div>
      </div>

      {/* Right Section - System Status */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">
            {new Date().toLocaleTimeString()}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <Wifi className="h-4 w-4 text-success" />
          <Badge variant="outline" className="text-success border-success">
            Connected
          </Badge>
        </div>
      </div>
    </div>
  );
};