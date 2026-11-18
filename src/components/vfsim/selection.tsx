import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { api } from "@/services/api";
import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

interface SelectionValue {
  selectedTxId: string;
  setSelectedTxId: (id: string) => void;
  riskThreshold: number;
  setRiskThreshold: (v: number) => void;
}

const SelectionContext = createContext<SelectionValue | null>(null);

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [selectedTxId, setSelectedTxId] = useState("");
  const [riskThreshold, setRiskThreshold] = useState(0.6);
  const txs = useQuery({queryKey:["transactions"],queryFn:api.getTransactions,refetchInterval:30_000});
  useEffect(()=>{
    if (!txs.data) return;
    if (txs.data.length===0) {if (selectedTxId) setSelectedTxId("");return;}
    if (!txs.data.some(tx=>tx.id===selectedTxId)) {
      const latest = txs.data[0];
      if (latest) setSelectedTxId(latest.id);
    }
  },[txs.data,selectedTxId]);
  const value = useMemo(
    () => ({ selectedTxId, setSelectedTxId, riskThreshold, setRiskThreshold }),
    [selectedTxId, riskThreshold],
  );
  return <SelectionContext.Provider value={value}>{children}</SelectionContext.Provider>;
}

export function useSelection() {
  const ctx = useContext(SelectionContext);
  if (!ctx) throw new Error("useSelection must be used inside SelectionProvider");
  return ctx;
}
