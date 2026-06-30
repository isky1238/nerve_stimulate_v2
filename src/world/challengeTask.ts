import { WorldAction } from "../core/arbitration";
import { SeededRandom } from "../core/random";
import { GridPosition, SensoryMapping, WorldObject, WorldObjectKind, WorldState } from "./world2d";

export type ChallengeTerminalReason =
  | "continue"
  | "food-contact"
  | "toxin-contact"
  | "toxin-avoided"
  | "conflict"
  | "step-limit";

export interface ChallengeObservedObject {
  id: string;
  kind: WorldObjectKind;
  dx: number;
  dy: number;
  distance: number;
  dropped: boolean;
}

export interface ChallengeRawObservation {
  visibleObjects: ChallengeObservedObject[];
  droppedObjects: ChallengeObservedObject[];
}

export interface ChallengeScenario {
  id: string;
  seed: number;
  width: number;
  height: number;
  maxSteps: number;
  agentStart: GridPosition;
  objects: WorldObject[];
}

export interface ChallengeRewardResult {
  reward: number;
  terminalReason: ChallengeTerminalReason;
  terminal: boolean;
  success: boolean;
  distanceDelta: number;
}

export const CHALLENGE_WIDTH = 7;
export const CHALLENGE_HEIGHT = 7;
export const DEFAULT_CHALLENGE_MAX_STEPS = 12;
export const DEFAULT_TRAIN_SEEDS = [1, 2, 3, 4, 5];
export const DEFAULT_EVAL_SEEDS = [101, 102, 103, 104, 105];

export function createChallengeScenarios(seeds: number[], maxSteps = DEFAULT_CHALLENGE_MAX_STEPS): ChallengeScenario[] {
  return seeds.flatMap((seed) => {
    const rng = new SeededRandom(seed);
    const distance = 2 + rng.nextInt(2);

    return [
      createChallengeScenario(`seed-${seed}-food-left`, seed * 10 + 1, "food", "left", distance, maxSteps),
      createChallengeScenario(`seed-${seed}-food-right`, seed * 10 + 2, "food", "right", distance, maxSteps),
      createChallengeScenario(`seed-${seed}-toxin-left`, seed * 10 + 3, "toxin", "left", 2, maxSteps),
      createChallengeScenario(`seed-${seed}-toxin-right`, seed * 10 + 4, "toxin", "right", 2, maxSteps)
    ];
  });
}

export function blankChallengeScenario(seed = 0, maxSteps = DEFAULT_CHALLENGE_MAX_STEPS): ChallengeScenario {
  return {
    id: "challenge-blank",
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: centerPosition(),
    objects: []
  };
}

export function conflictChallengeScenario(seed = 0, maxSteps = DEFAULT_CHALLENGE_MAX_STEPS): ChallengeScenario {
  const center = centerPosition();

  return {
    id: "challenge-conflict",
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: center,
    objects: [
      { id: "food-left", kind: "food", position: { x: center.x - 1, y: center.y } },
      { id: "food-right", kind: "food", position: { x: center.x + 1, y: center.y } }
    ]
  };
}

export function sameActionCompositeChallengeScenario(
  seed = 0,
  maxSteps = DEFAULT_CHALLENGE_MAX_STEPS
): ChallengeScenario {
  const center = centerPosition();

  return {
    id: "challenge-composite-same-action",
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: center,
    objects: [
      { id: "food-left", kind: "food", position: { x: center.x - 2, y: center.y } },
      { id: "toxin-right", kind: "toxin", position: { x: center.x + 2, y: center.y } }
    ]
  };
}

export function createChallengeWorldState(scenario: ChallengeScenario): WorldState {
  return {
    width: scenario.width,
    height: scenario.height,
    step: 0,
    agent: {
      position: { ...scenario.agentStart }
    },
    objects: scenario.objects.map((object) => ({
      ...object,
      position: { ...object.position }
    }))
  };
}

export function observeChallengeWorld(
  state: WorldState,
  observationDropout: number,
  rng: SeededRandom
): ChallengeRawObservation {
  const visibleObjects: ChallengeObservedObject[] = [];
  const droppedObjects: ChallengeObservedObject[] = [];

  for (const object of state.objects) {
    const dx = object.position.x - state.agent.position.x;
    const dy = object.position.y - state.agent.position.y;
    const observed = {
      id: object.id,
      kind: object.kind,
      dx,
      dy,
      distance: Math.abs(dx) + Math.abs(dy),
      dropped: false
    };

    if (dx === 0) {
      continue;
    }

    if (observationDropout > 0 && rng.next() < observationDropout) {
      droppedObjects.push({ ...observed, dropped: true });
    } else {
      visibleObjects.push(observed);
    }
  }

  return {
    visibleObjects,
    droppedObjects
  };
}

export function mapChallengeObservationToSensors(observation: ChallengeRawObservation): SensoryMapping {
  const activeSensorIds: string[] = [];
  const sensorReasons: Record<string, string> = {};

  for (const object of observation.visibleObjects) {
    const side = object.dx < 0 ? "Left" : "Right";
    const sensorId = `${object.kind}${side}`;
    activeSensorIds.push(sensorId);
    sensorReasons[sensorId] = `${object.kind}:${side.toLowerCase()}:dx=${object.dx}:dy=${object.dy}:distance=${object.distance}`;
  }

  activeSensorIds.sort();
  return {
    activeSensorIds,
    sensorReasons
  };
}

export function expectedActionForChallengeState(state: WorldState): WorldAction {
  const votes = state.objects
    .map((object): WorldAction | null => {
      const dx = object.position.x - state.agent.position.x;

      if (dx === 0) {
        return "noop";
      }

      if (object.kind === "food") {
        return dx < 0 ? "left" : "right";
      }

      return dx < 0 ? "right" : "left";
    })
    .filter((action): action is WorldAction => action !== null && action !== "noop");
  const uniqueVotes = Array.from(new Set(votes));

  if (uniqueVotes.length === 0) {
    return "noop";
  }

  return uniqueVotes.length === 1 ? uniqueVotes[0] : "conflict";
}

export function reverseAction(action: WorldAction): WorldAction {
  if (action === "left") {
    return "right";
  }
  if (action === "right") {
    return "left";
  }
  return action;
}

export function stepChallengeWorld(state: WorldState, action: WorldAction): WorldState {
  const next: WorldState = {
    width: state.width,
    height: state.height,
    step: state.step + 1,
    agent: {
      position: { ...state.agent.position }
    },
    objects: state.objects.map((object) => ({
      ...object,
      position: { ...object.position }
    }))
  };

  if (action === "left") {
    next.agent.position.x = Math.max(0, next.agent.position.x - 1);
  }

  if (action === "right") {
    next.agent.position.x = Math.min(next.width - 1, next.agent.position.x + 1);
  }

  return next;
}

export function scoreChallengeStep(before: WorldState, after: WorldState, action: WorldAction): ChallengeRewardResult {
  const beforeTarget = nearestObject(before);
  const afterTarget = nearestObject(after);

  if (action === "conflict") {
    return {
      reward: -0.2,
      terminalReason: "conflict",
      terminal: true,
      success: false,
      distanceDelta: 0
    };
  }

  if (!beforeTarget || !afterTarget) {
    return {
      reward: 0,
      terminalReason: "continue",
      terminal: false,
      success: false,
      distanceDelta: 0
    };
  }

  const beforeDistance = objectDistance(before, beforeTarget);
  const afterDistance = objectDistance(after, afterTarget);
  const distanceDelta = beforeDistance - afterDistance;

  if (afterTarget.kind === "food" && afterDistance === 0) {
    return {
      reward: 1,
      terminalReason: "food-contact",
      terminal: true,
      success: true,
      distanceDelta
    };
  }

  if (afterTarget.kind === "toxin" && afterDistance === 0) {
    return {
      reward: -1,
      terminalReason: "toxin-contact",
      terminal: true,
      success: false,
      distanceDelta
    };
  }

  if (afterTarget.kind === "toxin" && afterDistance >= 3) {
    return {
      reward: 1,
      terminalReason: "toxin-avoided",
      terminal: true,
      success: true,
      distanceDelta
    };
  }

  return {
    reward: shapedReward(afterTarget.kind, distanceDelta, action),
    terminalReason: "continue",
    terminal: false,
    success: false,
    distanceDelta
  };
}

export function shuffleScenarios(scenarios: ChallengeScenario[], seed: number): ChallengeScenario[] {
  const rng = new SeededRandom(seed);
  const shuffled = [...scenarios];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = rng.nextInt(index + 1);
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}

function createChallengeScenario(
  id: string,
  seed: number,
  kind: WorldObjectKind,
  side: "left" | "right",
  distance: number,
  maxSteps: number
): ChallengeScenario {
  const center = centerPosition();

  return {
    id,
    seed,
    width: CHALLENGE_WIDTH,
    height: CHALLENGE_HEIGHT,
    maxSteps,
    agentStart: center,
    objects: [
      {
        id: `${kind}-${side}-${distance}`,
        kind,
        position: {
          x: side === "left" ? center.x - distance : center.x + distance,
          y: center.y
        }
      }
    ]
  };
}

function shapedReward(kind: WorldObjectKind, distanceDelta: number, action: WorldAction): number {
  if (action === "noop") {
    return 0;
  }

  if (kind === "food") {
    return distanceDelta > 0 ? 0.1 : -0.1;
  }

  return distanceDelta < 0 ? 0.1 : -0.1;
}

function nearestObject(state: WorldState): WorldObject | null {
  let nearest: WorldObject | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (const object of state.objects) {
    const distance = objectDistance(state, object);

    if (distance < nearestDistance) {
      nearest = object;
      nearestDistance = distance;
    }
  }

  return nearest;
}

function objectDistance(state: WorldState, object: WorldObject): number {
  return Math.abs(object.position.x - state.agent.position.x) + Math.abs(object.position.y - state.agent.position.y);
}

function centerPosition(): GridPosition {
  return {
    x: Math.floor(CHALLENGE_WIDTH / 2),
    y: Math.floor(CHALLENGE_HEIGHT / 2)
  };
}
