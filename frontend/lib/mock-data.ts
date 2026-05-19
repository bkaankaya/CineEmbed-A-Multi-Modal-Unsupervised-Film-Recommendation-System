export interface Film {
  id: number
  title: string
  year: number
  rating: number
  votes: string
  genres: string[]
  country: string
  duration: string
  language: string
  director: string
  cluster: number
  overview: string
  style: string[]
  plot: string[]
  time: string
  place: string
  posterColor: string
}

export const FILMS: Film[] = [
  {
    id: 1,
    title: "Inception",
    year: 2010,
    rating: 8.8,
    votes: "2.4M",
    genres: ["Sci-Fi", "Action", "Thriller"],
    country: "USA",
    duration: "148 min",
    language: "English",
    director: "Christopher Nolan",
    cluster: 12,
    overview:
      "A skilled thief, the absolute best in the dangerous art of extraction — stealing valuable secrets from deep within the subconscious during the dream state — is offered a chance to regain his old life as payment for a task considered to be impossible: inception, the implantation of another person's idea into a target's subconscious.",
    style: ["mind-bending", "nonlinear", "heist", "psychological", "cerebral"],
    plot: ["dream", "subconscious", "heist", "architect", "limbo", "totem", "extraction", "layers"],
    time: "2010s",
    place: "Paris, Tokyo, USA",
    posterColor: "#1a2a4a",
  },
  {
    id: 2,
    title: "Interstellar",
    year: 2014,
    rating: 8.7,
    votes: "1.9M",
    genres: ["Sci-Fi", "Drama", "Adventure"],
    country: "USA",
    duration: "169 min",
    language: "English",
    director: "Christopher Nolan",
    cluster: 12,
    overview:
      "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival. A former NASA pilot leads the mission to find a new habitable planet as Earth faces extinction from a devastating blight.",
    style: ["epic", "emotional", "scientific", "visionary", "intimate"],
    plot: ["space", "wormhole", "black hole", "relativity", "love", "sacrifice", "future", "farming"],
    time: "Near future",
    place: "Earth, Saturn, Space",
    posterColor: "#0a1a2e",
  },
  {
    id: 3,
    title: "The Matrix",
    year: 1999,
    rating: 8.7,
    votes: "1.8M",
    genres: ["Sci-Fi", "Action"],
    country: "USA",
    duration: "136 min",
    language: "English",
    director: "The Wachowskis",
    cluster: 7,
    overview:
      "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth — the life he knows is the elaborate deception of an evil cyber-intelligence.",
    style: ["groundbreaking", "philosophical", "cyberpunk", "action-packed", "dystopian"],
    plot: ["simulation", "hacker", "red pill", "chosen one", "machines", "resistance", "reality"],
    time: "1999 / 2199",
    place: "Mega City, Machine City",
    posterColor: "#0a1a0a",
  },
  {
    id: 4,
    title: "Shutter Island",
    year: 2010,
    rating: 8.1,
    votes: "1.2M",
    genres: ["Mystery", "Thriller", "Drama"],
    country: "USA",
    duration: "138 min",
    language: "English",
    director: "Martin Scorsese",
    cluster: 12,
    overview:
      "In 1954, a U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane and finds himself questioning his own sanity.",
    style: ["neo-noir", "paranoid", "unreliable narrator", "gothic", "twisty"],
    plot: ["asylum", "detective", "identity", "delusion", "island", "conspiracy", "memory"],
    time: "1950s",
    place: "Boston Harbor, Ashecliffe Hospital",
    posterColor: "#1a1010",
  },
  {
    id: 5,
    title: "Memento",
    year: 2000,
    rating: 8.4,
    votes: "1.2M",
    genres: ["Mystery", "Thriller"],
    country: "USA",
    duration: "113 min",
    language: "English",
    director: "Christopher Nolan",
    cluster: 12,
    overview:
      "A man with short-term memory loss attempts to track down his wife's murderer using notes he scrawls on his body and polaroid photographs.",
    style: ["nonlinear", "noir", "puzzle", "psychological", "fragmented"],
    plot: ["memory", "tattoos", "polaroid", "revenge", "identity", "amnesia", "manipulation"],
    time: "2000s",
    place: "Los Angeles, USA",
    posterColor: "#1a1a0a",
  },
  {
    id: 6,
    title: "Eternal Sunshine of the Spotless Mind",
    year: 2004,
    rating: 8.3,
    votes: "985K",
    genres: ["Drama", "Romance", "Sci-Fi"],
    country: "USA",
    duration: "108 min",
    language: "English",
    director: "Michel Gondry",
    cluster: 9,
    overview:
      "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories. But as Joel is unraveling the memories, he discovers that he still loves Clementine.",
    style: ["surreal", "romantic", "bittersweet", "experimental", "introspective"],
    plot: ["memory erasure", "love", "relationship", "nostalgia", "loss", "identity"],
    time: "2000s",
    place: "New York, USA",
    posterColor: "#1a0a2a",
  },
  {
    id: 7,
    title: "Ex Machina",
    year: 2014,
    rating: 7.7,
    votes: "680K",
    genres: ["Sci-Fi", "Drama", "Thriller"],
    country: "UK",
    duration: "108 min",
    language: "English",
    director: "Alex Garland",
    cluster: 7,
    overview:
      "A young programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities of a highly advanced humanoid A.I.",
    style: ["minimalist", "cerebral", "claustrophobic", "tense", "feminist"],
    plot: ["AI", "robot", "Turing test", "consciousness", "manipulation", "isolation"],
    time: "Near future",
    place: "Remote research facility, Norway",
    posterColor: "#0a2a1a",
  },
  {
    id: 8,
    title: "Black Swan",
    year: 2010,
    rating: 8.0,
    votes: "740K",
    genres: ["Drama", "Horror", "Thriller"],
    country: "USA",
    duration: "108 min",
    language: "English",
    director: "Darren Aronofsky",
    cluster: 5,
    overview:
      "A committed dancer wins the lead role in a production of Tchaikovsky's Swan Lake only to find herself struggling to maintain her sanity and identity as the dark side of her personality begins to emerge.",
    style: ["psychological", "body horror", "obsessive", "operatic", "hallucinatory"],
    plot: ["ballet", "identity", "duality", "perfection", "jealousy", "madness"],
    time: "2010s",
    place: "New York City, USA",
    posterColor: "#0a0a1a",
  },
]

export const SIMILAR_FILMS_MAP: Record<number, Film[]> = {
  1: [FILMS[1], FILMS[4], FILMS[3], FILMS[2], FILMS[5], FILMS[6], FILMS[7]],
  2: [FILMS[0], FILMS[2], FILMS[6], FILMS[4], FILMS[3], FILMS[7], FILMS[5]],
  3: [FILMS[6], FILMS[1], FILMS[0], FILMS[4], FILMS[7], FILMS[5], FILMS[3]],
  4: [FILMS[0], FILMS[4], FILMS[7], FILMS[5], FILMS[2], FILMS[1], FILMS[6]],
  5: [FILMS[0], FILMS[3], FILMS[5], FILMS[1], FILMS[2], FILMS[7], FILMS[6]],
  6: [FILMS[4], FILMS[7], FILMS[0], FILMS[3], FILMS[1], FILMS[5], FILMS[2]],
  7: [FILMS[2], FILMS[1], FILMS[0], FILMS[6], FILMS[5], FILMS[4], FILMS[3]],
  8: [FILMS[3], FILMS[5], FILMS[0], FILMS[4], FILMS[6], FILMS[1], FILMS[2]],
}
