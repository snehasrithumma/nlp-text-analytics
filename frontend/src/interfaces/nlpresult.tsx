export interface SpacyNLPResult {
    summary: string;
    sentiment: string;
    keywords: string[];
    entities: Entity[];
}
   
type Entity = {
    text: string;
    label: string;
    score: number;
};

type HuggEntity = {
    entity: string;
    word: string;
    score: number;
};


export interface HuggingFaceNLPResult {
    sentiment: { label: string, score: number };
    summary: string;
    entities: HuggEntity[];
    topics: {
        sequence: string;
        labels: string[];
        scores: number[];
    }
}

interface ClassifierResult {
    label: string;
    score: number;
}

export type ClassifierResults = ClassifierResult[];