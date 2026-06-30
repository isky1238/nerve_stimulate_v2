import { ActionDecision, WorldAction } from "../core/arbitration";
import { SensoryMapping, WorldObject, WorldState } from "./world2d";
import { CHALLENGE_HEIGHT, CHALLENGE_WIDTH, ChallengeRawObservation, ChallengeScenario } from "./challengeTask";

export const DEFAULT_COMPLEX_MAX_STEPS = 6;
export const COMPLEX_SPIKE_TICKS = 3;

export interface ComplexSensoryInput {
  mapping: SensoryMapping;
  fireDurations: Record<string, number>;
}

export function mapComplexObservationToSensors(observation: ChallengeRawObservation): ComplexSensoryInput {
  const activeSensorIds: string[] = [];
  const fireDurations: Record<string, number> = {};
  const sensorReasons: Record<string, string> = {};

  for (const object of observation.visibleObjects) {
    const side = object.dx < 0 ? "Left" : "Right";
    const sensorId = `${object.kind}${side}`;
    const duration = Math.max(1, COMPLEX_SPIKE_TICKS + 1 - object.distance);
    activeSensorIds.push(sensorId);
    fireDurations[sensorId] = duration;
    sensorReasons[sensorId] =
      `${object.kind}:${side.toLowerCase()}:dx=${object.dx}:dy=${object.dy}:distance=${object.distance}:fireTicks=${duration}`;
  }

  activeSensorIds.sort();
  return {
    mapping: { activeSensorIds, sensorReasons },
    fireDurations
  };
}

export function expectedActionForComplexState(state: WorldState): WorldAction {
  const objectsWithDx = state.objects.filter((object) => object.position.x !== state.agent.position.x);
  if (objectsWithDx.length === 0) {
    return "noop";
  }

  const foods = objectsWithDx
    .filter((object) => object.kind === "food")
    .map((object) => ({ object, distance: manhattan(state, object) }))
    .sort((a, b) => a.distance - b.distance);
  const toxins = objectsWithDx
    .filter((object) => object.kind === "toxin")
    .map((object) => ({ object, distance: manhattan(state, object) }))
    .sort((a, b) => a.distance - b.distance);

  const nearestFood = foods[0];
  const nearestToxin = toxins[0];

  const foodVote: WorldAction | null = nearestFood
    ? nearestFood.object.position.x < state.agent.position.x
      ? "left"
      : "right"
    : null;
  const toxinVote: WorldAction | null = nearestToxin
    ? nearestToxin.object.position.x > state.agent.position.x
      ? "left"
      : "right"
    : null;

  if (foodVote && toxinVote) {
    if (foodVote === toxinVote) {
      return foodVote;
    }
    if (nearestToxin.distance <= nearestFood.distance) {
      return toxinVote;
    }
    return foodVote;
  }

  if (foodVote) {
    if (foods.length >= 2) {
      const second = foods[1];
      const nearestSide = sideOf(nearestFood.object, state);
      const secondSide = sideOf(second.object, state);
      if (nearestSide !== secondSide && second.distance === nearestFood.distance) {
        return "conflict";
      }
    }
    return foodVote;
  }

  if (toxinVote) {
    if (toxins.length >= 2) {
      const second = toxins[1];
      const nearestSide = sideOf(nearestToxin.object, state);
      const secondSide = sideOf(second.object, state);
      if (nearestSide !== secondSide && second.distance === nearestToxin.distance) {
        return "conflict";
      }
    }
    return toxinVote;
  }

  return "noop";
}

export function arbitrateComplexMotorAction(spikeCounts: Record<string, number>): ActionDecision {
  const left = spikeCounts["leftMotor"] ?? 0;
  const right = spikeCounts["rightMotor"] ?? 0;
  const activeMotors: string[] = [];
  if (left > 0) {
    activeMotors.push("leftMotor");
  }
  if (right > 0) {
    activeMotors.push("rightMotor");
  }
  activeMotors.sort();

  if (left === 0 && right === 0) {
    return {
      action: "noop",
      activeMotors: [],
      mappedActions: [],
      reason: "no-active-motor"
    };
  }

  if (left > right) {
    return {
      action: "left",
      activeMotors,
      mappedActions: ["left"],
      reason: "spike-count-left"
    };
  }

  if (right > left) {
    return {
      action: "right",
      activeMotors,
      mappedActions: ["right"],
      reason: "spike-count-right"
    };
  }

  return {
    action: "conflict",
    activeMotors,
    mappedActions: ["left", "right"],
    reason: "equal-spike-count"
  };
}

export function compositeSameDirectionScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  return [
    {
      id: "complex-composite-food-left-toxin-right",
      seed: 201,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left", kind: "food", position: { x: center.x - 2, y: center.y } },
        { id: "toxin-right", kind: "toxin", position: { x: center.x + 2, y: center.y } }
      ]
    },
    {
      id: "complex-composite-food-right-toxin-left",
      seed: 202,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-right", kind: "food", position: { x: center.x + 2, y: center.y } },
        { id: "toxin-left", kind: "toxin", position: { x: center.x - 2, y: center.y } }
      ]
    }
  ];
}

export function distractorScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  return [
    {
      id: "complex-distractor-food-left-near-food-right-far",
      seed: 211,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left-near", kind: "food", position: { x: center.x - 1, y: center.y } },
        { id: "food-right-far", kind: "food", position: { x: center.x + 3, y: center.y } }
      ]
    },
    {
      id: "complex-distractor-food-right-near-food-left-far",
      seed: 212,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-right-near", kind: "food", position: { x: center.x + 1, y: center.y } },
        { id: "food-left-far", kind: "food", position: { x: center.x - 3, y: center.y } }
      ]
    },
    {
      id: "complex-distractor-toxin-left-near-toxin-right-far",
      seed: 213,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "toxin-left-near", kind: "toxin", position: { x: center.x - 1, y: center.y } },
        { id: "toxin-right-far", kind: "toxin", position: { x: center.x + 3, y: center.y } }
      ]
    },
    {
      id: "complex-distractor-toxin-right-near-toxin-left-far",
      seed: 214,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "toxin-right-near", kind: "toxin", position: { x: center.x + 1, y: center.y } },
        { id: "toxin-left-far", kind: "toxin", position: { x: center.x - 3, y: center.y } }
      ]
    }
  ];
}

export function priorityScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  return [
    {
      id: "complex-priority-food-left-near-toxin-left-far",
      seed: 221,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left-near", kind: "food", position: { x: center.x - 2, y: center.y } },
        { id: "toxin-left-far", kind: "toxin", position: { x: center.x - 3, y: center.y } }
      ]
    },
    {
      id: "complex-priority-food-left-far-toxin-left-near",
      seed: 222,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left-far", kind: "food", position: { x: center.x - 3, y: center.y } },
        { id: "toxin-left-near", kind: "toxin", position: { x: center.x - 2, y: center.y } }
      ]
    },
    {
      id: "complex-priority-food-right-near-toxin-right-far",
      seed: 223,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-right-near", kind: "food", position: { x: center.x + 2, y: center.y } },
        { id: "toxin-right-far", kind: "toxin", position: { x: center.x + 3, y: center.y } }
      ]
    },
    {
      id: "complex-priority-food-right-far-toxin-right-near",
      seed: 224,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-right-far", kind: "food", position: { x: center.x + 3, y: center.y } },
        { id: "toxin-right-near", kind: "toxin", position: { x: center.x + 2, y: center.y } }
      ]
    }
  ];
}

export function trueConflictScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  return [
    {
      id: "complex-conflict-food-left-food-right-equidistant",
      seed: 231,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "food-left", kind: "food", position: { x: center.x - 2, y: center.y } },
        { id: "food-right", kind: "food", position: { x: center.x + 2, y: center.y } }
      ]
    },
    {
      id: "complex-conflict-toxin-left-toxin-right-equidistant",
      seed: 232,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: "toxin-left", kind: "toxin", position: { x: center.x - 2, y: center.y } },
        { id: "toxin-right", kind: "toxin", position: { x: center.x + 2, y: center.y } }
      ]
    }
  ];
}

export function semanticConflictScenarios(maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario[] {
  const center = centerPosition();
  const semanticScenario = (
    id: string,
    seed: number,
    side: "left" | "right",
    distance: number
  ): ChallengeScenario => {
    const offset = side === "left" ? -distance : distance;
    return {
      id,
      seed,
      width: CHALLENGE_WIDTH,
      height: CHALLENGE_HEIGHT,
      maxSteps,
      agentStart: { ...center },
      objects: [
        { id: `toxin-${side}-d${distance}`, kind: "toxin", position: { x: center.x + offset, y: center.y } },
        { id: `food-${side}-d${distance}`, kind: "food", position: { x: center.x + offset, y: center.y } }
      ]
    };
  };

  return [
    semanticScenario("complex-semantic-conflict-food-left-toxin-left-d2", 241, "left", 2),
    semanticScenario("complex-semantic-conflict-food-right-toxin-right-d2", 242, "right", 2),
    semanticScenario("complex-semantic-conflict-food-left-toxin-left-d1", 243, "left", 1),
    semanticScenario("complex-semantic-conflict-food-right-toxin-right-d1", 244, "right", 1),
    semanticScenario("complex-semantic-conflict-food-left-toxin-left-d3", 245, "left", 3),
    semanticScenario("complex-semantic-conflict-food-right-toxin-right-d3", 246, "right", 3)
  ];
}

export function blankComplexScenario(seed = 0, maxSteps = DEFAULT_COMPLEX_MAX_STEPS): ChallengeScenario {
  return {
    id: "complex-blank",
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: centerPosition(),
    objects: []
  };
}

function sideOf(object: WorldObject, state: WorldState): "left" | "right" {
  return object.position.x < state.agent.position.x ? "left" : "right";
}

function manhattan(state: WorldState, object: WorldObject): number {
  return Math.abs(object.position.x - state.agent.position.x) + Math.abs(object.position.y - state.agent.position.y);
}

function centerPosition() {
  return {
    x: Math.floor(CHALLENGE_WIDTH / 2),
    y: Math.floor(CHALLENGE_HEIGHT / 2)
  };
}
