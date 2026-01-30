type NavigatorConnection = {
  effectiveType?: string;
  type?: string;
  rtt?: number;
  downlink?: number;
  saveData?: boolean;
};

type NavigatorWithConnection = Navigator & {
  connection?: NavigatorConnection;
  mozConnection?: NavigatorConnection;
  webkitConnection?: NavigatorConnection;
};

export type NetworkInformation = {
  effectiveConnectionType: string;
  connectionType: number;
  rtt?: number;
  downlink?: number;
  saveData?: boolean;
};

const CONNECTION_TYPE_MAP: Record<string, number> = {
  unknown: 0,
  other: 0,
  bluetooth: 1,
  cellular: 2,
  ethernet: 3,
  wifi: 4,
  wimax: 5,
  none: 6,
};

export const getNetworkInformation = (): NetworkInformation => {
  if (typeof navigator === "undefined") {
    return { effectiveConnectionType: "UNKNOWN", connectionType: 0 };
  }

  const nav = navigator as NavigatorWithConnection;
  const connection = nav.connection ?? nav.mozConnection ?? nav.webkitConnection;
  if (!connection) {
    return { effectiveConnectionType: "UNKNOWN", connectionType: 0 };
  }

  const effectiveConnectionType = connection.effectiveType
    ? connection.effectiveType.toUpperCase()
    : "UNKNOWN";
  const typeValue = connection.type ? connection.type.toLowerCase() : "unknown";
  const connectionType = CONNECTION_TYPE_MAP[typeValue] ?? 0;

  return {
    effectiveConnectionType,
    connectionType,
    rtt: typeof connection.rtt === "number" ? connection.rtt : undefined,
    downlink: typeof connection.downlink === "number" ? connection.downlink : undefined,
    saveData: typeof connection.saveData === "boolean" ? connection.saveData : undefined,
  };
};
