export type WorldAction = "left" | "right" | "noop" | "conflict";

export interface MotorActionMapping {
  motorId: string;
  action: Exclude<WorldAction, "noop" | "conflict">;
}

export interface ActionDecision {
  action: WorldAction;
  activeMotors: string[];
  mappedActions: string[];
  reason: string;
}

const DEFAULT_MOTOR_ACTIONS: MotorActionMapping[] = [
  { motorId: "leftMotor", action: "left" },
  { motorId: "rightMotor", action: "right" }
];

export function targetMotorForAction(action: WorldAction): string | null {
  if (action === "left") {
    return "leftMotor";
  }

  if (action === "right") {
    return "rightMotor";
  }

  return null;
}

export function arbitrateMotorAction(
  activeMotorIds: string[],
  mappings: MotorActionMapping[] = DEFAULT_MOTOR_ACTIONS
): ActionDecision {
  const uniqueMotorIds = Array.from(new Set(activeMotorIds)).sort();
  const mappedActions = uniqueMotorIds
    .map((motorId) => mappings.find((mapping) => mapping.motorId === motorId)?.action)
    .filter((action): action is Exclude<WorldAction, "noop" | "conflict"> => action !== undefined);
  const uniqueActions = Array.from(new Set(mappedActions)).sort();

  if (uniqueMotorIds.length === 0) {
    return {
      action: "noop",
      activeMotors: [],
      mappedActions: [],
      reason: "no-active-motor"
    };
  }

  if (mappedActions.length !== uniqueMotorIds.length) {
    return {
      action: "conflict",
      activeMotors: uniqueMotorIds,
      mappedActions: uniqueActions,
      reason: "unknown-active-motor"
    };
  }

  if (uniqueActions.length === 1 && uniqueMotorIds.length === 1) {
    return {
      action: uniqueActions[0],
      activeMotors: uniqueMotorIds,
      mappedActions: uniqueActions,
      reason: "single-motor"
    };
  }

  return {
    action: "conflict",
    activeMotors: uniqueMotorIds,
    mappedActions: uniqueActions,
    reason: "multiple-motors"
  };
}
