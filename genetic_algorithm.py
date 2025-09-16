from flask import Flask, request, jsonify
import random
import numpy as np
from typing import List, Dict, Any
import json

app = Flask(__name__)

class TourismGeneticAlgorithm:
    def __init__(self, population_size=50, generations=100, mutation_rate=0.05, crossover_rate=0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def converter_horario(self, horario_int: int) -> int:
        """Converte horário de formato HHMM para minutos desde 00:00"""
        horas = horario_int // 100
        minutos = horario_int % 100
        return horas * 60 + minutos
    
    def calcular_tempo_total_disponivel(self, perfil: Dict) -> int:
        """Calcula o tempo total disponível em minutos"""
        inicio = self.converter_horario(perfil.get('horarioInicio', 800))
        fim = self.converter_horario(perfil.get('horarioFinal', 1800))
        return fim - inicio
    
    def calcular_duracao_roteiro(self, individual: List[int], locais: List[Dict]) -> int:
        """Calcula a duração total do roteiro em minutos"""
        duracao_total = 0
        
        for i, local_idx in enumerate(individual):
            if local_idx < len(locais):
                local = locais[local_idx]
                duracao_local = local.get('tempoMedioVisita', 60)
                duracao_total += duracao_local
                
                if i < len(individual) - 1:
                    tempo_deslocamento = local.get('tempoDeslocamento', 15)
                    duracao_total += tempo_deslocamento
        return duracao_total
    
    def parsear_horario_funcionamento(self, horario_str: str) -> List[tuple]:
        """
        Parseia string de horário de funcionamento
        Ex: "08:00-12:00, 13:30-17:30" -> [(480, 720), (810, 1050)]
        """
        if not horario_str or horario_str.lower() in ['conforme programação', 'conforme eventos']:
            return [(0, 1440)]
            
        periodos = []
        try:
            for periodo in horario_str.split(','):
                periodo = periodo.strip()
                if '-' in periodo:
                    inicio_str, fim_str = periodo.split('-')
                    inicio_minutos = self.converter_hora_texto_para_minutos(inicio_str.strip())
                    fim_minutos = self.converter_hora_texto_para_minutos(fim_str.strip())
                    periodos.append((inicio_minutos, fim_minutos))
        except:
            periodos = [(0, 1440)]
            
        return periodos if periodos else [(0, 1440)]
    
    def converter_hora_texto_para_minutos(self, hora_str: str) -> int:
        """Converte hora em texto para minutos. Ex: '08:30' -> 510"""
        try:
            horas, minutos = hora_str.split(':')
            return int(horas) * 60 + int(minutos)
        except:
            return 0
        
    def calculate_fitness(self, individual: List[int], locais: List[Dict], perfil: Dict) -> float:
        """Calcula o fitness de um indivíduo (roteiro)"""
        if not individual:
            return 0.0
            
        fitness = 0.0
        tags_usuario = set(perfil.get('tagsPreferencias', []))
        tempo_disponivel = self.calcular_tempo_total_disponivel(perfil)
        duracao_roteiro = self.calcular_duracao_roteiro(individual, locais)
        
        # Penalização severa se o roteiro exceder o tempo disponível
        if duracao_roteiro > tempo_disponivel:
            excesso = duracao_roteiro - tempo_disponivel
            fitness -= excesso * 2
            
        # Penalização leve se o roteiro for muito curto (menos de 70% do tempo)
        tempo_minimo = tempo_disponivel * 0.7
        if duracao_roteiro < tempo_minimo:
            deficit = tempo_minimo - duracao_roteiro
            fitness -= deficit * 0.5
            
        # Bônus por otimização do tempo (usar entre 85-95% do tempo disponível)
        if 0.85 * tempo_disponivel <= duracao_roteiro <= 0.95 * tempo_disponivel:
            fitness += 100
        
        # Pontuação base por local (como antes)
        for local_idx in individual:
            if local_idx < len(locais):
                local = locais[local_idx]
                
                # Pontuação por avaliação média
                fitness += local.get('avaliacaoMedia', 0) * 10
                
                # Pontuação por tags correspondentes
                tags_local = set(local.get('tags', []))
                tags_match = len(tags_usuario.intersection(tags_local))
                fitness += tags_match * 20
                
                # Penaliza preços muito altos
                preco = local.get('precoMedio', 0)
                if preco > 100:
                    fitness -= (preco - 100) * 0.1
                    
                # Bônus para locais adequados para crianças se necessário
                if perfil.get('adequadoParaCriancas', False) and local.get('adequadoCriancas', False):
                    fitness += 15
                    
                # Penaliza ambientes noturnos se deve evitar
                if perfil.get('evitarAmbienteNoturno', False) and local.get('ambienteNoturno', False):
                    fitness -= 30
                    
        # Penaliza roteiros muito curtos em número de locais
        if len(individual) < 2:
            fitness -= 50
            
        return max(0, fitness)
            
    def local_esta_aberto(self, local: Dict, horario_minutos: int) -> bool:
        """Verifica se o local está aberto no horário especificado"""
        horario_funcionamento = local.get('horarioFuncionamento', '')
        
        if not horario_funcionamento:
            return True
            
        periodos = self.parsear_horario_funcionamento(horario_funcionamento)
        
        for inicio, fim in periodos:
            if inicio <= horario_minutos <= fim:
                return True
                
        return False
    
    def roteiro_respeita_horarios(self, individual: List[int], locais: List[Dict], perfil: Dict) -> bool:
        """Verifica se o roteiro respeita os horários de funcionamento dos locais"""
        inicio_minutos = self.converter_horario(perfil.get('horarioInicio', 800))
        tempo_atual = inicio_minutos
        
        for local_idx in individual:
            if local_idx < len(locais):
                local = locais[local_idx]
                
                if not self.local_esta_aberto(local, tempo_atual):
                    return False
                
                duracao_local = local.get('tempoMedioVisita', 60)
                tempo_saida = tempo_atual + duracao_local
                
                if not self.local_esta_aberto(local, tempo_saida):
                    return False
                
                tempo_atual = tempo_saida + local.get('tempoDeslocamento', 15)
                
        return True
    
    def create_individual(self, locais: List[Dict], perfil: Dict) -> List[int]:
        """Cria um indivíduo (roteiro) respeitando tempo e horários de funcionamento"""
        tempo_disponivel = self.calcular_tempo_total_disponivel(perfil)
        inicio_minutos = self.converter_horario(perfil.get('horarioInicio', 800))
        
        individual = []
        tempo_atual = inicio_minutos
        locais_disponiveis = list(range(len(locais)))
        random.shuffle(locais_disponiveis)
        
        for local_idx in locais_disponiveis:
            local = locais[local_idx]
            
            if not self.local_esta_aberto(local, tempo_atual):
                continue
                
            duracao_local = local.get('tempoMedioVisita', 60)
            tempo_saida = tempo_atual + duracao_local
            
            if not self.local_esta_aberto(local, tempo_saida):
                continue
                
            tempo_deslocamento = local.get('tempoDeslocamento', 15) if individual else 0
            tempo_total_necessario = duracao_local + tempo_deslocamento
            
            if (tempo_atual - inicio_minutos) + tempo_total_necessario <= tempo_disponivel:
                individual.append(local_idx)
                tempo_atual = tempo_saida + (local.get('tempoDeslocamento', 15) if len(individual) > 1 else 0)
            
        if not individual and locais:
            for local_idx in locais_disponiveis:
                local = locais[local_idx]
                if self.local_esta_aberto(local, inicio_minutos):
                    individual.append(local_idx)
                    break
                    
        return individual
    
    def create_population(self, locais: List[Dict], perfil: Dict) -> List[List[int]]:
        """Cria a população inicial"""
        return [self.create_individual(locais, perfil) for _ in range(self.population_size)]
    
    def tournament_selection(self, population: List[List[int]], fitnesses: List[float], 
                             tournament_size: int = 3) -> List[int]:
        """Seleção por torneio"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: List[int], parent2: List[int], locais: List[Dict], perfil: Dict) -> List[int]:
        """Crossover entre dois pais respeitando tempo e horários"""
        if len(parent1) < 1 or len(parent2) < 1:
            return parent1 if len(parent1) >= len(parent2) else parent2
            
        tempo_disponivel = self.calcular_tempo_total_disponivel(perfil)
        inicio_minutos = self.converter_horario(perfil.get('horarioInicio', 800))
        all_elements = list(set(parent1) | set(parent2))
        random.shuffle(all_elements)
        
        child = []
        tempo_atual = inicio_minutos
        
        for local_idx in all_elements:
            if local_idx < len(locais):
                local = locais[local_idx]
                
                if not self.local_esta_aberto(local, tempo_atual):
                    continue
                
                duracao_local = local.get('tempoMedioVisita', 60)
                tempo_saida = tempo_atual + duracao_local
                
                if not self.local_esta_aberto(local, tempo_saida):
                    continue
                
                tempo_deslocamento = local.get('tempoDeslocamento', 15) if child else 0
                tempo_total_necessario = duracao_local + tempo_deslocamento
                
                if (tempo_atual - inicio_minutos) + tempo_total_necessario <= tempo_disponivel:
                    child.append(local_idx)
                    tempo_atual = tempo_saida + (local.get('tempoDeslocamento', 15) if len(child) > 1 else 0)
                    
        return child if child else [all_elements[0]] if all_elements else []
    
    def mutate(self, individual: List[int], locais: List[Dict], perfil: Dict) -> List[int]:
        """Mutação de um indivíduo respeitando tempo e horários"""
        if random.random() > self.mutation_rate:
            return individual
            
        mutated = individual.copy()
        tempo_disponivel = self.calcular_tempo_total_disponivel(perfil)
        
        mutation_type = random.choice(['add', 'remove', 'replace'])
        
        if mutation_type == 'add':
            available = [i for i in range(len(locais)) if i not in mutated]
            if available:
                candidate = random.choice(available)
                test_individual = mutated + [candidate]
                
                if (self.calcular_duracao_roteiro(test_individual, locais) <= tempo_disponivel and 
                    self.roteiro_respeita_horarios(test_individual, locais, perfil)):
                    mutated.append(candidate)
                    
        elif mutation_type == 'remove' and len(mutated) > 1:
            mutated.remove(random.choice(mutated))
            
        elif mutation_type == 'replace' and mutated:
            available = [i for i in range(len(locais)) if i not in mutated]
            if available:
                old_idx = random.randint(0, len(mutated) - 1)
                candidate = random.choice(available)
                
                test_individual = mutated.copy()
                test_individual[old_idx] = candidate
                
                if (self.calcular_duracao_roteiro(test_individual, locais) <= tempo_disponivel and 
                    self.roteiro_respeita_horarios(test_individual, locais, perfil)):
                    mutated[old_idx] = candidate
                    
        return mutated
    
    def optimize(self, locais: List[Dict], perfil: Dict) -> List[Dict]:
        """Executa o algoritmo genético"""
        if not locais:
            return []
            
        tempo_disponivel = self.calcular_tempo_total_disponivel(perfil)
        print(f"Tempo disponível: {tempo_disponivel} minutos ({tempo_disponivel//60}h{tempo_disponivel%60}min)")
        
        # Cria população inicial
        population = self.create_population(locais, perfil)
        
        for generation in range(self.generations):
            # Calcula fitness
            fitnesses = [self.calculate_fitness(ind, locais, perfil) for ind in population]
            
            # Cria nova população
            new_population = []
            
            # Elitismo: mantém o melhor
            best_idx = fitnesses.index(max(fitnesses))
            new_population.append(population[best_idx].copy())
            
            # ALTERAÇÃO: Usa o novo loop de geração de população com taxa de crossover
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover ou replicação baseada na taxa
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2, locais, perfil)
                else:
                    # Replica o melhor dos dois pais
                    if self.calculate_fitness(parent1, locais, perfil) > self.calculate_fitness(parent2, locais, perfil):
                        child = parent1.copy()
                    else:
                        child = parent2.copy()
                
                child = self.mutate(child, locais, perfil)
                
                new_population.append(child)
                
            population = new_population
        
        # Retorna melhor solução
        final_fitnesses = [self.calculate_fitness(ind, locais, perfil) for ind in population]
        best_individual = population[final_fitnesses.index(max(final_fitnesses))]
        
        result = []
        duracao_total = 0
        inicio_minutos = self.converter_horario(perfil.get('horarioInicio', 800))
        tempo_atual = inicio_minutos
        
        for i, idx in enumerate(best_individual):
            if idx < len(locais):
                local = locais[idx].copy()
                # Pontuação individual simples:
                fitness = 0.0
                tags_usuario = set(perfil.get('tagsPreferencias', []))
                tags_local = set(local.get('tags', []))
                fitness += local.get('avaliacaoMedia', 0) * 10
                fitness += len(tags_usuario.intersection(tags_local)) * 20
                local['pontuacaoRecomendacao'] = fitness

                duracao_local = local.get('tempoMedioVisita', 60)
                local['duracaoRealMinutos'] = duracao_local
                
                local['horaChegadaEstimada'] = self.converter_minutos_para_hora(tempo_atual)
                
                tempo_saida = tempo_atual + duracao_local
                local['horaSaidaEstimada'] = self.converter_minutos_para_hora(tempo_saida)
                
                local['horarioFuncionamentoOriginal'] = local.get('horarioFuncionamento', 'Não informado')
                
                periodos = self.parsear_horario_funcionamento(local.get('horarioFuncionamento', ''))
                if periodos:
                    horarios_formatados = []
                    for inicio, fim in periodos:
                        inicio_str = self.converter_minutos_para_hora(inicio)
                        fim_str = self.converter_minutos_para_hora(fim)
                        horarios_formatados.append(f"{inicio_str}-{fim_str}")
                    local['horarioFuncionamentoFormatado'] = ', '.join(horarios_formatados)
                else:
                    local['horarioFuncionamentoFormatado'] = 'Sempre aberto'
                
                if i < len(best_individual) - 1:
                    tempo_deslocamento = local.get('tempoDeslocamento', 15)
                    local['tempoDeslocamento'] = tempo_deslocamento
                    duracao_total += tempo_deslocamento
                    tempo_atual = tempo_saida + tempo_deslocamento
                else:
                    tempo_atual = tempo_saida
                
                duracao_total += duracao_local
                result.append(local)
        
        print(f"Roteiro gerado com {len(result)} locais e duração total de {duracao_total} minutos")
        
        return result
    
    def converter_minutos_para_hora(self, minutos: int) -> str:
        """Converte minutos para formato HH:MM"""
        horas = (minutos // 60) % 24
        mins = minutos % 60
        return f"{horas:02d}:{mins:02d}"

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
            
        # ALTERAÇÃO: Passando o novo parâmetro de crossover_rate para o construtor
        ga = TourismGeneticAlgorithm(
            population_size=50,
            generations=100,
            mutation_rate=0.1,
            crossover_rate=0.8  # Usando 80% como valor padrão
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