# The Deepwell Chronicles — A Test Corpus for FFN Context

# Each section simulates a conversation turn (~200-500 tokens each).
# Total: ~5,000 tokens across 15 turns.
# Contains 40+ verifiable facts (names, numbers, places, events).

STORY_TURNS = [
    {
        "turn": 1,
        "text": """The city of Deepwell was founded in 2041 by urban architect Miriam Castillo on the floor of an abandoned copper mine in northern Chile. The mine shaft, originally excavated by Corporacion Nacional del Cobre in 1978, descends exactly 847 meters below the Atacama surface. Castillo's original proposal, submitted to the Chilean government on March 3rd 2038, called for a population of 12,000 residents living in concentric ring structures carved from the mine walls. The project was funded by a consortium of 7 nations led by Norway, with a total construction budget of 4.2 billion euros. Construction began on September 15th 2039 and the first residents moved in on June 1st 2041.""",
        "facts": [
            ("Who founded Deepwell?", ["miriam", "castillo"]),
            ("How deep is Deepwell?", ["847"]),
            ("When did construction of Deepwell begin?", ["september", "2039"]),
            ("How many nations funded Deepwell?", ["7", "seven"]),
            ("What was the construction budget?", ["4.2", "billion"]),
        ]
    },
    {
        "turn": 2,
        "text": """The power infrastructure of Deepwell relies on three redundant systems. The primary source is a geothermal tap reaching 3,200 meters below the city floor into a magma chamber discovered by geologist Henrik Larsen in 2036. This generates 340 megawatts continuously. The secondary system consists of 14,000 square meters of concentrated solar collectors on the surface, producing 85 megawatts during daylight. The tertiary backup is a thorium molten salt reactor designed by physicist Yuna Takahashi, rated at 120 megawatts. Total peak capacity is 545 megawatts for a city that typically draws 280 megawatts. The excess is sold to the Chilean national grid via a 47-kilometer transmission line to the coastal city of Antofagasta.""",
        "facts": [
            ("How deep is the geothermal tap?", ["3200", "3,200"]),
            ("Who discovered the magma chamber?", ["henrik", "larsen"]),
            ("How many megawatts does geothermal produce?", ["340"]),
            ("Who designed the thorium reactor?", ["yuna", "takahashi"]),
            ("What is the total peak capacity?", ["545"]),
            ("How far is the transmission line to Antofagasta?", ["47"]),
        ]
    },
    {
        "turn": 3,
        "text": """Deepwell's agriculture is managed by Chief Botanist Oluwaseun Adeyemi, who developed the Vertical Cascade system. The system uses 23 tiered hydroponic levels spiraling down the mine walls, each level spanning 1,400 square meters. The primary crops are modified quinoa (strain QW-7, developed at Universidad de Chile), dwarf avocados, and three varieties of mushroom including the bioluminescent Mycena deepwellensis, a species discovered in the mine walls and found nowhere else on Earth. The total agricultural output feeds 94 percent of the city's population. The remaining 6 percent is imported via a weekly supply train running on the original mine rail system, which has been upgraded to maglev by TransAndes Corporation.""",
        "facts": [
            ("Who is the Chief Botanist?", ["oluwaseun", "adeyemi"]),
            ("How many hydroponic levels?", ["23"]),
            ("What quinoa strain is used?", ["qw-7"]),
            ("What percentage of food is locally grown?", ["94"]),
            ("What unique mushroom was discovered?", ["mycena", "deepwellensis"]),
        ]
    },
    {
        "turn": 4,
        "text": """The governance of Deepwell operates under the Charter of Subterranean Rights, drafted by constitutional lawyer Fatima Al-Rashid in 2040. The charter establishes a Council of Twelve, with 4 members elected by residents, 4 appointed by the founding consortium, and 4 selected by lottery from the general population every 18 months. The current Council President is Dr. Kofi Mensah, a former neurosurgeon from Accra who has served since January 2049. All council meetings are held in the Obsidian Chamber, a natural basalt cavern at the 400-meter level that seats exactly 200 observers. Decisions require a two-thirds majority of 8 votes.""",
        "facts": [
            ("Who drafted the Charter of Subterranean Rights?", ["fatima", "al-rashid"]),
            ("How many council members are there?", ["twelve", "12"]),
            ("Who is the current Council President?", ["kofi", "mensah"]),
            ("At what depth is the Obsidian Chamber?", ["400"]),
            ("How many votes for a decision?", ["8", "eight"]),
        ]
    },
    {
        "turn": 5,
        "text": """Transportation within Deepwell uses a network of 6 vertical express elevators and 31 horizontal tram lines. The main elevator, nicknamed the Plunge, drops from surface to the bottom level in 73 seconds at a maximum speed of 42 kilometers per hour. It was manufactured by ThyssenKrupp Elevator and installed in 2040. The horizontal trams are autonomous electric vehicles running on dedicated tracks carved into the ring walls, each capable of carrying 28 passengers. The entire transit system was designed by transportation engineer Priya Chakrabarti, who previously designed the Mumbai Metro Phase 4. Average commute time in Deepwell is 11 minutes.""",
        "facts": [
            ("How many vertical elevators?", ["6", "six"]),
            ("How fast does the Plunge descend?", ["73", "seconds"]),
            ("Who designed the transit system?", ["priya", "chakrabarti"]),
            ("How many passengers per tram?", ["28"]),
            ("What is the average commute time?", ["11"]),
        ]
    },
    {
        "turn": 6,
        "text": """Deepwell's research district occupies levels 600 through 720, known as the Quiet Zone due to the exceptional seismic stability at that depth. The flagship facility is the Larsen Neutrino Observatory, a 40-meter spherical chamber filled with 50,000 liters of ultra-pure heavy water, detecting an average of 847 neutrino events per day. The observatory is directed by particle physicist Dr. Sven Eriksson, who relocated from CERN in 2045. Adjacent to the observatory is the Deepwell Quantum Computing Center, housing a 4,096-qubit processor cooled to 15 millikelvin. The center's lead researcher, Dr. Chen Wei, holds patents on 3 novel error-correction algorithms.""",
        "facts": [
            ("What levels is the research district on?", ["600", "720"]),
            ("How many liters of heavy water in the observatory?", ["50000", "50,000"]),
            ("Who directs the neutrino observatory?", ["sven", "eriksson"]),
            ("How many qubits in the quantum processor?", ["4096", "4,096"]),
            ("How cold is the quantum processor?", ["15", "millikelvin"]),
        ]
    },
    {
        "turn": 7,
        "text": """The cultural heart of Deepwell is the Resonance Hall, a concert venue carved from a natural crystal geode at the 500-meter level. The hall has perfect acoustics with a reverberation time of exactly 2.1 seconds, comparable to the Vienna Musikverein. It seats 1,800 people and hosts the Deepwell Philharmonic, founded by conductor Alejandra Ruiz in 2043. The orchestra has 67 members and performs 48 concerts annually. The city also maintains the Subterranean Museum of Geological Art, curated by Dr. Ingrid Hoffman, which houses 2,340 mineral specimens including the largest natural amethyst crystal ever found, weighing 847 kilograms — the same number as the city's depth in meters, a coincidence the residents consider lucky.""",
        "facts": [
            ("What is the reverberation time of Resonance Hall?", ["2.1"]),
            ("Who founded the Deepwell Philharmonic?", ["alejandra", "ruiz"]),
            ("How many orchestra members?", ["67"]),
            ("How many mineral specimens in the museum?", ["2340", "2,340"]),
            ("How heavy is the largest amethyst?", ["847"]),
        ]
    },
    {
        "turn": 8,
        "text": """Water in Deepwell comes from an underground aquifer discovered at 1,100 meters depth. The aquifer, named the Castillo Reservoir after the city's founder, holds an estimated 2.3 billion liters and is replenished by glacial melt from the Andes at a rate of 14 million liters per year. Water is pumped upward through 4 primary risers and purified at the Level 300 treatment facility managed by hydrologist Dr. James Okafor. The treatment plant processes 8.4 million liters daily and achieves a purity rating of 99.97 percent, exceeding WHO standards. Wastewater is recycled through a closed-loop bioreactor system designed by environmental engineer Sofia Petrov, achieving 96 percent water recovery.""",
        "facts": [
            ("How deep is the Castillo Reservoir?", ["1100", "1,100"]),
            ("How many liters does it hold?", ["2.3", "billion"]),
            ("Who manages the treatment facility?", ["james", "okafor"]),
            ("What is the daily processing volume?", ["8.4"]),
            ("What is the water recovery rate?", ["96"]),
        ]
    },
]

# Questions spanning ALL turns for the needle-in-haystack test
ALL_QUESTIONS = []
for turn in STORY_TURNS:
    ALL_QUESTIONS.extend(turn["facts"])
