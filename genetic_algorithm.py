from flask import Flask, request, jsonify
import random
import numpy as np
from typing import List, Dict, Any
import json

app = Flask(__name__)

class TourismGeneticAlgorithm:
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.max_itinerary_size = 8  # Máximo de locais no roteiro
        
    def calculate_fitness(self, individual: List[int], locais: List[Dict], perfil: Dict) -> float:
        """Calcula o fitness de um indivíduo (roteiro)"""
        if not individual:
            return 0.0
            
        fitness = 0.0
        tags_usuario = set(perfil.get('tagsPreferencias', []))
        
        # Pontuação base por local
        for local_idx in individual:
            if local_idx < len(locais):
                local = locais[local_idx]
                
                # Pontuação por avaliação média
                fitness += local.get('avaliacaoMedia', 0) * 10
                
                # Pontuação por tags correspondentes
                tags_local = set(local.get('tags', []))
                tags_match = len(tags_usuario.intersection(tags_local))
                fitness += tags_match * 20
                
                # Penaliza preços muito altos se não especificado
                preco = local.get('precoMedio', 0)
                if preco > 100:
                    fitness -= (preco - 100) * 0.1
                    
                # Bônus para locais adequados para crianças se necessário
                if perfil.get('adequadoParaCriancas', False) and local.get('adequadoCriancas', False):
                    fitness += 15
                    
                # Penaliza ambientes noturnos se deve evitar
                if perfil.get('evitarAmbienteNoturno', False) and local.get('ambienteNoturno', False):
                    fitness -= 30
                    
        # Penaliza roteiros muito curtos ou muito longos
        if len(individual) < 3:
            fitness -= 50
        elif len(individual) > self.max_itinerary_size:
            fitness -= 20 * (len(individual) - self.max_itinerary_size)
            
        return max(0, fitness)
    
    def create_individual(self, num_locais: int) -> List[int]:
        """Cria um indivíduo (roteiro) aleatório"""
        size = random.randint(3, min(self.max_itinerary_size, num_locais))
        return random.sample(range(num_locais), size)
    
    def create_population(self, num_locais: int) -> List[List[int]]:
        """Cria a população inicial"""
        return [self.create_individual(num_locais) for _ in range(self.population_size)]
    
    def tournament_selection(self, population: List[List[int]], fitnesses: List[float], 
                           tournament_size: int = 3) -> List[int]:
        """Seleção por torneio"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Crossover entre dois pais"""
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1 if len(parent1) >= len(parent2) else parent2
            
        # Pega elementos únicos dos dois pais
        common_elements = list(set(parent1) & set(parent2))
        all_elements = list(set(parent1) | set(parent2))
        
        # Cria filho com elementos comuns + alguns aleatórios
        child_size = random.randint(3, min(self.max_itinerary_size, len(all_elements)))
        
        if len(common_elements) >= child_size:
            return random.sample(common_elements, child_size)
        else:
            child = common_elements.copy()
            remaining = [x for x in all_elements if x not in child]
            child.extend(random.sample(remaining, min(child_size - len(child), len(remaining))))
            return child
    
    def mutate(self, individual: List[int], num_locais: int) -> List[int]:
        """Mutação de um indivíduo"""
        if random.random() > self.mutation_rate:
            return individual
            
        mutated = individual.copy()
        
        mutation_type = random.choice(['add', 'remove', 'replace'])
        
        if mutation_type == 'add' and len(mutated) < self.max_itinerary_size:
            available = [i for i in range(num_locais) if i not in mutated]
            if available:
                mutated.append(random.choice(available))
                
        elif mutation_type == 'remove' and len(mutated) > 3:
            mutated.remove(random.choice(mutated))
            
        elif mutation_type == 'replace' and mutated:
            available = [i for i in range(num_locais) if i not in mutated]
            if available:
                old_idx = random.randint(0, len(mutated) - 1)
                mutated[old_idx] = random.choice(available)
                
        return mutated
    
    def optimize(self, locais: List[Dict], perfil: Dict) -> List[Dict]:
        """Executa o algoritmo genético"""
        num_locais = len(locais)
        if num_locais == 0:
            return []
            
        # Cria população inicial
        population = self.create_population(num_locais)
        
        for generation in range(self.generations):
            # Calcula fitness
            fitnesses = [self.calculate_fitness(ind, locais, perfil) for ind in population]
            
            # Cria nova população
            new_population = []
            
            # Elitismo: mantém o melhor
            best_idx = fitnesses.index(max(fitnesses))
            new_population.append(population[best_idx].copy())
            
            # Gera resto da população
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, num_locais)
                
                new_population.append(child)
                
            population = new_population
        
        # Retorna melhor solução
        final_fitnesses = [self.calculate_fitness(ind, locais, perfil) for ind in population]
        best_individual = population[final_fitnesses.index(max(final_fitnesses))]
        
        # Converte para formato de resposta
        result = []
        for idx in best_individual:
            if idx < len(locais):
                local = locais[idx].copy()
                local['pontuacaoRecomendacao'] = self.calculate_fitness([idx], locais, perfil)
                result.append(local)
                
        # Ordena por pontuação
        result.sort(key=lambda x: x['pontuacaoRecomendacao'], reverse=True)
        
        return result[:8]  # Máximo 8 locais

@app.route('/api/recomendar', methods=['POST'])
def recomendar():
    try:
        data = request.get_json()
        
        if not data or 'locais' not in data or 'perfil' not in data:
            return jsonify({'error': 'Dados inválidos'}), 400
            
        locais = data['locais']
        perfil = data['perfil']
        
        if not locais:
            return jsonify([]), 200
            
        # Executa algoritmo genético
        ga = TourismGeneticAlgorithm(
            population_size=50,
            generations=100,
            mutation_rate=0.1
        )
        
        recomendacoes = ga.optimize(locais, perfil)
        
        return jsonify(recomendacoes), 200
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK', 'service': 'Genetic Algorithm Tourism API'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)