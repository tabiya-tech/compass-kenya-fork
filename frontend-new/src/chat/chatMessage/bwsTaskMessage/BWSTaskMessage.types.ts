export interface BWSAlternative {
  wa_id: string;
  label: string;
}

export interface BWSTaskMessageProps {
  taskId: string;
  taskNumber: number;
  totalTasks: number;
  alternatives: BWSAlternative[];
  // Resolves to true when the submission succeeded; on false the card is re-enabled so the user can retry
  onSubmit: (taskId: string, bestWaId: string, worstWaId: string) => Promise<boolean>;
}

// Shape of the metadata field sent from backend on BWS_TASK messages
export interface BWSTaskMetadata {
  task_id: string;
  task_number: number;
  total_tasks: number;
  alternatives: BWSAlternative[];
}
