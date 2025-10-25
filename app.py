import random
import json
import gradio as gr

# ===============================
# 🔹 Base de Datos de Personajes
# ===============================

characters = [
    {
        "id": 1,
        "name": "León-O",
        "universe": "ThunderCats",
        "type": "Felino-humanoide",
        "special_attack": "Espada del Augurio – 'Ho!'",
        "prompt": {
            "description": "Cuerpo felino de 2.15m, musculatura potente, pelaje naranja con manchas.",
            "costume": "Armadura metálica sobre piel.",
            "pose": "Pose de batalla con la Espada del Augurio, gritando 'Ho!'",
            "environment": "Ruinas post-apocalípticas con neblina volumétrica.",
            "technical_specs": {
                "camera": "ARRI Alexa 65, 50mm, f/1.4, ISO-800",
                "lighting": "3-point lights con contrastes fuertes",
                "render": "Unreal Engine 5.3, Lumen GI, ray-tracing"
            }
        }
    },
    # Sailor Moon universe
    {
        "id": 10,
        "name": "Sailor Moon (Usagi Tsukino)",
        "universe": "Sailor Moon",
        "type": "humana (cosplay)",
        "special_attack": "Moon Tiara Action",
        "prompt": {
          "description": "cuerpo humano perfectamente modelado, 1.75-1.90m, cabello largo con coletas, musculatura y piel detallada",
          "costume": "cosplay altamente fiel con materiales realistas, costuras visibles, accesorios exactos",
          "pose": "pose de combate dinámica con expresión determinada, lista para Moon Tiara Action",
          "environment": "entorno post-apocalíptico urbano con ruinas, humo denso, chispas eléctricas, luz volumétrica neón",
          "technical_specs": {
            "camera": "Hasselblad X1D, 45mm, f/2.0, ISO-640",
            "lighting": "luz de luna con relleno de neón, alto contraste",
            "render": "Unreal Engine 5.3, 8K UHD, HDR"
          }
        }
    },
    {
        "id": 11,
        "name": "Sailor Mercury (Ami Mizuno)",
        "universe": "Sailor Moon",
        "type": "humana (cosplay)",
        "special_attack": "Mercury Aqua Mist",
        "prompt": {
          "description": "joven altamente inteligente con cabello corto azul, mirada calmada",
          "costume": "uniforme de Sailor con textura de tela realista y detalles metálicos",
          "pose": "pose con niebla de agua envolviendo el entorno",
          "environment": "escenario acuático con luces frías y bruma sutil",
          "technical_specs": {
            "camera": "Sony A7R IV, 55mm, f/1.8, ISO-500",
            "lighting": "luz suave azulada",
            "render": "Unreal Engine 5, ray-tracing"
          }
        }
    },
    {
        "id": 12,
        "name": "Sailor Mars (Rei Hino)",
        "universe": "Sailor Moon",
        "type": "humana (cosplay)",
        "special_attack": "Fire Soul",
        "prompt": {
          "description": "mujer fuerte con cabello largo negro, expresión determinada",
          "costume": "uniforme rojo y blanco con texturas brillantes",
          "pose": "pose con bola de fuego en manos",
          "environment": "fondo oscuro con llamas y humo",
          "technical_specs": {
            "camera": "Canon EOS R6, 85mm, f/1.4, ISO-640",
            "lighting": "iluminación cálida y dramática",
            "render": "Unreal Engine 5.3, Lumen GI"
          }
        }
    },
    {
        "id": 13,
        "name": "Sailor Jupiter (Makoto Kino)",
        "universe": "Sailor Moon",
        "type": "humana (cosplay)",
        "special_attack": "Supreme Thunder",
        "prompt": {
          "description": "mujer atlética con cabello castaño recogido, mirada poderosa",
          "costume": "uniforme verde con detalles metálicos y texturas realistas",
          "pose": "pose de combate invocando rayos y truenos",
          "environment": "cielo tormentoso con relámpagos",
          "technical_specs": {
            "camera": "Nikon D850, 50mm, f/1.8, ISO-400",
            "lighting": "iluminación atmosférica con efectos de luz eléctrica",
            "render": "Unreal Engine 5.3, ray-tracing"
          }
        }
    },
    {
        "id": 14,
        "name": "Sailor Venus (Minako Aino)",
        "universe": "Sailor Moon",
        "type": "humana (cosplay)",
        "special_attack": "Venus Love-Me Chain",
        "prompt": {
          "description": "mujer con cabello rubio largo, expresión alegre y confiada",
          "costume": "uniforme naranja estilizado con detalles brillantes",
          "pose": "pose lanzando cadena mágica",
          "environment": "fondo luminoso con corazones de energía",
          "technical_specs": {
            "camera": "Fujifilm GFX 100S, 110mm, f/2, ISO-320",
            "lighting": "luz brillante y colores vivos",
            "render": "Unreal Engine 5, high detail textures"
          }
        }
    },
    # Street Fighter universe
    {
        "id": 20,
        "name": "Ryu",
        "universe": "Street Fighter",
        "type": "humano (cosplay)",
        "special_attack": "Hadouken",
        "prompt": {
          "description": "luchador masculino, 1.75-1.90m, cabello corto y desordenado, musculatura definida y rasgos faciales marcados",
          "costume": "gi blanco realista con textura de tela auténtica y guantes rojos",
          "pose": "pose clásica de Hadouken, con energía visible en manos y expresión decidida",
          "environment": "ruinas urbanas post-apocalípticas con humo, chispas eléctricas y luz volumétrica",
          "technical_specs": {
            "camera": "Sony A1 Cine, 35mm f/1.8, ISO-400",
            "lighting": "iluminación dramática con sombras fuertes",
            "render": "Unreal Engine 5, ray-tracing de alta precisión"
          }
        }
    },
    {
        "id": 21,
        "name": "Ken",
        "universe": "Street Fighter",
        "type": "humano (cosplay)",
        "special_attack": "Shoryuken",
        "prompt": {
          "description": "luchador atlético con cabello rubio, musculatura marcada",
          "costume": "gi rojo realista con detalles desgastados y guantes blancos",
          "pose": "pose explosiva de puñetazo ascendente con fuego en mano",
          "environment": "campo de batalla con chispas y fuego ambiente",
          "technical_specs": {
            "camera": "Canon EOS R5, 50mm, f/1.4, ISO-500",
            "lighting": "iluminación cálida y dramática",
            "render": "Unreal Engine 5.3, Lumen GI"
          }
        }
    },
    {
        "id": 22,
        "name": "Chun-Li",
        "universe": "Street Fighter",
        "type": "humana (cosplay)",
        "special_attack": "Lightning Kicks",
        "prompt": {
          "description": "mujer atlética, cabello en moño con blusa azul ajustada y botas altas",
          "costume": "uniforme azul con detalles dorados y textura realista",
          "pose": "pose dinámica de patadas veloces con efecto de electricidad",
          "environment": "plaza urbana iluminada, luces de neón",
          "technical_specs": {
            "camera": "Nikon Z7 II, 85mm, f/1.8, ISO-640",
            "lighting": "iluminación fría con reflejos azules",
            "render": "Unreal Engine 5, ray-tracing"
          }
        }
    },
    {
        "id": 23,
        "name": "Guile",
        "universe": "Street Fighter",
        "type": "humano (cosplay)",
        "special_attack": "Sonic Boom",
        "prompt": {
          "description": "soldado robusto con cabello en estilo militar, musculoso y mirada concentrada",
          "costume": "uniforme militar verde, botas tácticas",
          "pose": "pose de disparo de onda sónica",
          "environment": "base militar destruida con luces frías",
          "technical_specs": {
            "camera": "Sony A7 III, 35mm, f/1.8, ISO-320",
            "lighting": "iluminación dura y dramática",
            "render": "Unreal Engine 5.3, Lumen GI"
          }
        }
    },
    {
        "id": 24,
        "name": "M. Bison",
        "universe": "Street Fighter",
        "type": "villano",
        "special_attack": "Psycho Crusher",
        "prompt": {
          "description": "líder villano, atuendo rojo militar, capa negra, mirada amenazante",
          "costume": "uniforme rojo satinado con texturas detalladas y capa oscura",
          "pose": "pose agresiva, embestida de energía psíquica",
          "environment": "base oscura con energía roja pulsante",
          "technical_specs": {
            "camera": "Canon EOS-1D X Mark III, 85mm, f/1.2, ISO-800",
            "lighting": "iluminación roja dramática",
            "render": "Unreal Engine 5, ray-tracing"
          }
        }
    },
    # The King of Fighters universe
    {
        "id": 30,
        "name": "Kyo Kusanagi",
        "universe": "The King of Fighters",
        "type": "humano",
        "special_attack": "Orochinagi",
        "prompt": {
          "description": "luchador masculino, cabello negro, ropa moderna con chaqueta negra, rodeado de llamas ardientes, pose combativa",
          "costume": "tela y piel con reflejos realistas",
          "pose": "pose dinámica con efectos visuales de fuego",
          "environment": "ciudad nocturna con luces de neón",
          "technical_specs": {
            "camera": "Canon EOS R5 Cine RAW, Canon RF 85mm f/1.2L USM lens",
            "lighting": "resplandor de fuego",
            "render": "Ultra HD 16K, color grading DaVinci Resolve"
          }
        }
    },
    {
        "id": 31,
        "name": "Iori Yagami",
        "universe": "The King of Fighters",
        "type": "humano demoníaco",
        "special_attack": "Ya Otome",
        "prompt": {
          "description": "luchador masculino con cabello púrpura salvaje, expresión feroz, ropa rota y oscura, envuelto en fuego púrpura",
          "costume": "tela rasgada y piel con texturas ricas",
          "pose": "pose agresiva en fantasía oscura",
          "environment": "calles destruidas en noche lluviosa",
          "technical_specs": {
            "camera": "Canon EOS R5 Cine RAW",
            "lighting": "iluminación oscura y brillosa",
            "render": "color grading DaVinci Resolve"
          }
        }
    },
    {
        "id": 32,
        "name": "Terry Bogard",
        "universe": "The King of Fighters",
        "type": "humano",
        "special_attack": "Buster Wolf",
        "prompt": {
          "description": "luchador musculoso con gorra roja, chaleco sin mangas rojo, pose confiada y enérgica",
          "costume": "tela y piel con reflejos realistas",
          "pose": "pose dinámica y enérgica",
          "environment": "arena de lucha al aire libre",
          "technical_specs": {
            "camera": "Canon EOS R5, 50mm, f/1.8, ISO-640",
            "lighting": "luz natural brillante",
            "render": "Unreal Engine 5.3, 8K UHD"
          }
        }
    },
    {
        "id": 33,
        "name": "Leona Heidern",
        "universe": "The King of Fighters",
        "type": "militar humano",
        "special_attack": "V-Slasher",
        "prompt": {
          "description": "soldado femenina con cabello azul, uniforme táctico oscuro, expresión seria y mirada fija",
          "costume": "tela táctica y armadura ligera",
          "pose": "pose agresiva y determinada",
          "environment": "zona de combate en jungla",
          "technical_specs": {
            "camera": "ARRI Alexa 65, 50mm, f/1.4, ISO-800",
            "lighting": "iluminación dramática",
            "render": "Unreal Engine 5.3, ray-tracing"
          }
        }
    },
    {
        "id": 34,
        "name": "Mai Shiranui",
        "universe": "The King of Fighters",
        "type": "humano",
        "special_attack": "Chou Hissatsu Shinobi Bachi",
        "prompt": {
          "description": "luchadora femenina con cabello largo pelirrojo, vestimenta tradicional japonesa, pose seductora",
          "costume": "tela ligera con detalles bordados, texturas realistas",
          "pose": "pose con abanicos y fuego en movimiento",
          "environment": "templo tradicional japonés con luces cálidas",
          "technical_specs": {
            "camera": "Canon EOS R6, 85mm, f/1.6, ISO-500",
            "lighting": "iluminación suave y natural",
            "render": "Unreal Engine 5, Lumen GI"
          }
        }
    },
    # Metal Slug universe
    {
        "id": 40,
        "name": "Marco Rossi",
        "universe": "Metal Slug",
        "type": "soldado humano",
        "special_attack": "Tiro certero",
        "prompt": {
          "description": "soldado valiente con uniforme militar en camuflaje, expresión determinada y postura lista para combate",
          "costume": "uniforme militar verde detallado, botas tácticas, casco ligero",
          "pose": "pose dinámica disparando su rifle de asalto",
          "environment": "campo de batalla con explosiones y humo",
          "technical_specs": {
            "camera": "Canon EOS 5D Mark IV, 50mm, f/1.8, ISO-800",
            "lighting": "iluminación dura intensa, sombras fuertes",
            "render": "Unreal Engine 5.3, ray-tracing, texturas 32-bit"
          }
        }
    },
    {
        "id": 41,
        "name": "Tarma Roving",
        "universe": "Metal Slug",
        "type": "soldado humano",
        "special_attack": "Granada explosiva",
        "prompt": {
          "description": "soldado con bandana y gafas oscuras, expresión confiada y postura defensiva",
          "costume": "uniforme táctico oscuro, chaleco antibalas",
          "pose": "pose apuntando con lanzagranadas",
          "environment": "ruinas incendiadas de ciudad",
          "technical_specs": {
            "camera": "Nikon D850, 35mm, f/1.8, ISO-640",
            "lighting": "luz natural con sombras definidas",
            "render": "Unreal Engine 5, Lumen GI"
          }
        }
    },
    {
        "id": 42,
        "name": "Eri Kasamoto",
        "universe": "Metal Slug",
        "type": "soldado humano",
        "special_attack": "Ataque con ametralladora",
        "prompt": {
          "description": "soldado femenina con cabello corto, uniforme de camuflaje y expresión enfocada",
          "costume": "uniforme de camuflaje con chaleco táctico",
          "pose": "pose disparando ametralladora con precisión",
          "environment": "jungla densa en zona de combate",
          "technical_specs": {
            "camera": "Sony A7R IV, 85mm, f/1.4, ISO-500",
            "lighting": "iluminación suave y ambiental",
            "render": "Unreal Engine 5, ray-tracing"
          }
        }
    },
    {
        "id": 43,
        "name": "Fio Germi",
        "universe": "Metal Slug",
        "type": "soldado humano",
        "special_attack": "Disparo de francotirador",
        "prompt": {
          "description": "soldado femenina con cabello largo recogido y vestimenta táctica moderna",
          "costume": "uniforme táctico con accesorios modernos e texturas detalladas",
          "pose": "pose apuntando con rifle de francotirador",
          "environment": "calle urbana en guerra con escombros",
          "technical_specs": {
            "camera": "Canon EOS R6, 70mm, f/2.0, ISO-640",
            "lighting": "luz dramática tenue con contrastes",
            "render": "Unreal Engine 5.3, Lumen GI"
          }
        }
    },
    {
        "id": 44,
        "name": "Ralf Jones",
        "universe": "Metal Slug",
        "type": "soldado humano",
        "special_attack": "Puñetazo poderoso",
        "prompt": {
          "description": "soldado musculoso, cabeza rapada, chaleco táctico y expresión confiada",
          "costume": "uniforme táctico con texturas realistas y detalles metálicos",
          "pose": "pose de combate con puñetazo fuerte",
          "environment": "desierto hostil con cielo despejado",
          "technical_specs": {
            "camera": "Sony A1, 50mm, f/1.2, ISO-400",
            "lighting": "iluminación natural brillante",
            "render": "Unreal Engine 5, ray-tracing de alta precisión"
          }
        }
    },
    # Darkstalkers universe
    {
        "id": 50,
        "name": "Demitri Maximoff",
        "universe": "Darkstalkers",
        "type": "vampiro",
        "special_attack": "Midnight Bliss",
        "prompt": {
          "description": "vampiro noble de Rumania, cabello oscuro, mirada imponente y elegante",
          "costume": "vestimenta oscura con capa roja y detalles góticos",
          "pose": "pose elegante y amenazante, envolviendo enemigos en su aura oscura",
          "environment": "castillo oscuro con luna llena y niebla densa",
          "technical_specs": {
            "camera": "ARRI Alexa, 85mm, f/1.2, ISO-800",
            "lighting": "iluminación baja con tonos rojos",
            "render": "Unreal Engine 5 con ray-tracing y Lumen GI"
          }
        }
    },
    {
        "id": 51,
        "name": "Morrigan Aensland",
        "universe": "Darkstalkers",
        "type": "súcubo",
        "special_attack": "Finishing Shower",
        "prompt": {
          "description": "súcubo seductora con cabello verde oscuro, ojos brillantes",
          "costume": "atuendo ajustado con alas y detalles demoníacos",
          "pose": "pose seductora lanzando misiles mágicos",
          "environment": "mundo demoníaco con realismo oscuro y neón",
          "technical_specs": {
            "camera": "Canon EOS R5, 50mm, f/1.4, ISO-640",
            "lighting": "iluminación dramática con contrastes fuertes",
            "render": "Unreal Engine 5.3, texturas 32-bit"
          }
        }
    },
    {
        "id": 52,
        "name": "Felicia",
        "universe": "Darkstalkers",
        "type": "catgirl",
        "special_attack": "Rolling Buckler",
        "prompt": {
          "description": "felina humana, pelaje blanco-azul, expresión alegre y traviesa",
          "costume": "traje adaptado con texturas de pelaje realista",
          "pose": "pose activa rodando con garras extendidas",
          "environment": "escenario nocturno con luces tenues",
          "technical_specs": {
            "camera": "Sony A7R IV, 85mm, f/1.8, ISO-800",
            "lighting": "iluminación suave y cálida",
            "render": "Unreal Engine 5, ray-tracing"
          }
        }
    },
    {
        "id": 53,
        "name": "Jon Talbain",
        "universe": "Darkstalkers",
        "type": "hombre lobo",
        "special_attack": "Beast Cannon",
        "prompt": {
          "description": "hombre lobo musculoso con pelaje gris, expresión feroz",
          "costume": "vestimenta rasgada acorde a su naturaleza salvaje",
          "pose": "pose agresiva con embestidas rápidas",
          "environment": "bosque oscuro con niebla espesa",
          "technical_specs": {
            "camera": "Nikon Z9, 70mm, f/2, ISO-500",
            "lighting": "iluminación dramática con sombras fuertes",
            "render": "Unreal Engine 5, iluminación global"
          }
        }
    },
    {
        "id": 54,
        "name": "Lord Raptor",
        "universe": "Darkstalkers",
        "type": "zombi guitarrista",
        "special_attack": "Death Voltage",
        "prompt": {
          "description": "zombi con chaqueta negra y guitarra eléctrica, expresión macabra",
          "costume": "ropa de rocker, cuero y metal oxidado",
          "pose": "pose energizada tocando guitarra con rayos eléctricos",
          "environment": "escenario oscuro con luces eléctricas",
          "technical_specs": {
            "camera": "Canon EOS R6, 24mm, f/1.4, ISO-800",
            "lighting": "iluminación intensa y puntual",
            "render": "Unreal Engine 5, texturas con detalle alto"
          }
        }
    },
    # ThunderCats universe
    {
        "id": 60,
        "name": "Tigro",
        "universe": "ThunderCats",
        "type": "felino-humanoide",
        "special_attack": "Bololátigo",
        "prompt": {
          "description": "felino ágil, cuerpo atlético y musculoso, pelaje rayado",
          "costume": "ropa ligera para máxima movilidad",
          "pose": "pose de ataque con bololátigo extendido",
          "environment": "ruinas urbanas con iluminación de neón",
          "technical_specs": {
            "camera": "Canon EOS R5, 85mm, f/1.4, ISO-640",
            "lighting": "iluminación urbana con neón y sombras marcadas",
            "render": "Unreal Engine 5, ray-tracing"
          }
        }
    },
    {
        "id": 61,
        "name": "Chitara",
        "universe": "ThunderCats",
        "type": "felino-humanoide",
        "special_attack": "Boomerang Supersónico",
        "prompt": {
          "description": "felina esbelta con pelo corto, mirada intensa",
          "costume": "uniforme táctico estilizado, materiales tecnológicos",
          "pose": "pose con bastón giratorio de alto impacto",
          "environment": "zona de combate con humo y chispas",
          "technical_specs": {
            "camera": "Hasselblad X1D, 50mm, f/2, ISO-800",
            "lighting": "foco difuso y luz dramática",
            "render": "Unreal Engine 5.3, Lumen GI"
          }
        }
    },
    {
        "id": 62,
        "name": "Pantro",
        "universe": "ThunderCats",
        "type": "felino-humanoide",
        "special_attack": "Nunchakus Electrificados",
        "prompt": {
          "description": "felino musculoso con brazos cibernéticos, piel con marcas de batalla",
          "costume": "armadura táctica con detalles metálicos y desgaste visible",
          "pose": "pose de combate con nunchakus cargados",
          "environment": "campo de batalla rocoso con niebla",
          "technical_specs": {
            "camera": "Sony A7 III, 35mm, f/1.8, ISO-400",
            "lighting": "iluminación dura y contrastante",
            "render": "Unreal Engine 5, ray-tracing"
          }
        }
    },
    {
        "id": 63,
        "name": "Jaga",
        "universe": "ThunderCats",
        "type": "felino-humanoide",
        "special_attack": "Sabiduría Ancestral",
        "prompt": {
          "description": "anciano felino sabio, aspecto noble y tranquilo",
          "costume": "vestiduras tradicionales con símbolos antiguos",
          "pose": "pose espiritual, emanando energía mística",
          "environment": "ruinas antiguas con luz tenue y mística",
          "technical_specs": {
            "camera": "Canon EOS R6, 50mm, f/1.8, ISO-320",
            "lighting": "iluminación espiritual suave",
            "render": "Unreal Engine 5, Lumen GI"
          }
        }
    },
    {
        "id": 64,
        "name": "Snarf",
        "universe": "ThunderCats",
        "type": "criatura felina pequeña",
        "special_attack": "Defensa y Apoyo",
        "prompt": {
          "description": "pequeña criatura felina con pelaje naranja y blanco, expresión cariñosa",
          "costume": "pelaje detallado ultra realista",
          "pose": "pose de alerta y protectora",
          "environment": "interior base ThunderCats, iluminación cálida",
          "technical_specs": {
            "camera": "Sony A7R IV, 85mm, f/2, ISO-640",
            "lighting": "luz cálida y difusa",
            "render": "Unreal Engine 5, ray-tracing"
          }
        }
    }
]

# ===============================
# 🔹 Funciones
# ===============================

def random_character_prompt():
    """
    Retorna un prompt aleatorio basado en la lista de personajes.
    """
    chara = random.choice(characters)

    tech_specs = f"- Cámara: {chara['prompt']['technical_specs']['camera']}\n- Render: {chara['prompt']['technical_specs']['render']}"
    if 'lighting' in chara['prompt']['technical_specs']:
        tech_specs += f"\n- Iluminación: {chara['prompt']['technical_specs']['lighting']}"

    prompt = f"""
### 🎭 {chara['name']} ({chara['universe']})
**Tipo:** {chara['type']}
**Ataque Especial:** {chara['special_attack']}

**Descripción:** {chara['prompt']['description']}
**Costume:** {chara['prompt']['costume']}
**Pose:** {chara['prompt']['pose']}
**Environment:** {chara['prompt']['environment']}

🔧 **Specs Técnicos**
{tech_specs}
"""
    return prompt.strip()

def catalog_view(universe):
    """
    Retorna la ficha de todos los personajes de un universo.
    """
    filtered = [ch for ch in characters if ch["universe"] == universe] if universe != "Todos" else characters
    if not filtered:
        return "No hay personajes en este universo."

    out = "## 📖 Catálogo de Personajes\n\n"
    for ch in filtered:
        out += f"""
---
### 🎭 {ch['name']} ({ch['universe']})
- **Tipo:** {ch['type']}
- **Ataque Especial:** {ch['special_attack']}
- **Descripción:** {ch['prompt']['description']}
- **Costume:** {ch['prompt']['costume']}
- **Escenario:** {ch['prompt']['environment']}
"""
    return out

# ===============================
# 🔹 Interfaz en Gradio
# ===============================
def launch_app():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="red", secondary_hue="blue")) as demo:
        gr.Markdown("# 🦾 BATUTO-VERSE Prompt Generator")
        gr.Markdown("Genera prompts hiperrealistas de personajes icónicos 🎲")

        with gr.Row():
            random_btn = gr.Button("🎲 Generar Aleatorio", variant="primary")
            universes = ["Todos"] + sorted(list(set([ch["universe"] for ch in characters])))
            universe_dropdown = gr.Dropdown(choices=universes,
                                            label="📚 Filtrar por Universo", value="Todos")
            catalog_btn = gr.Button("📖 Ver Catálogo")

        output = gr.Markdown("")

        random_btn.click(fn=random_character_prompt, inputs=None, outputs=output)
        catalog_btn.click(fn=catalog_view, inputs=universe_dropdown, outputs=output)

    demo.launch(share=True)

# Ejecutar en Colab/HF
if __name__ == "__main__":
    launch_app()
